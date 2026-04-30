"""
LADS-Causal框架主运行脚本

时滞感知双层因果根因定位框架的主入口。
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
import random

# 添加路径以便导入causelens模块

import torch as th  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]

from config import LADSConfig, default_config
from data_preprocessor import TrainTicketDataPreprocessor, AIOps2025DataPreprocessor, RCAbenchDataPreprocessor
from evaluator import evaluate_lads_results
from lag_aware_rca import lag_aware_dual_stage_rca
try:
    from stage2_reranker_split_eval import run_oof_cv_from_csv
except Exception:
    run_oof_cv_from_csv = None
from log import Logger  # pyright: ignore[reportMissingImports]

logger = Logger(__name__)

# 仓库根目录：.../CauseLens-master
REPO_ROOT = Path(__file__).resolve().parent

def set_all_seeds(seed: int) -> None:
    """统一设置随机种子，保证不预训练/不训练时的可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)

    # CPU 场景也设置一下，避免后续扩展 GPU 时出现不可复现
    try:
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False
    except Exception:
        pass

    # DGL（如果可用）
    try:
        import dgl  # pyright: ignore[reportMissingImports]
        # DGL 不同版本函数名不同，这里做兼容
        if hasattr(dgl, "seed"):
            dgl.seed(seed)
        elif hasattr(dgl, "random") and hasattr(dgl.random, "seed"):
            dgl.random.seed(seed)
    except Exception:
        pass


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='RippleRCA: lag-aware dual-stage root cause analysis'
    )

    parser.add_argument('--dataset_type', type=str,
                       choices=['trainticket', 'aiops2025', 'rcabench'],
                       default='rcabench',
                       help='Dataset type: trainticket / aiops2025 / rcabench')
    parser.add_argument('--data_dir', type=str,
                       default=None,
                       help='Dataset directory. If omitted, use the default path for the selected dataset.')
    parser.add_argument('--train_dates', nargs='+',
                       default=['2024-04-05', '2024-04-06'],
                       help='Training dates')
    parser.add_argument('--test_dates', nargs='+',
                       default=['2024-04-05', '2024-04-06'],
                       help='Test dates')
    parser.add_argument('--rcabench_scenarios',
                       nargs='+',
                       default=None,
                       help='Optional RCAbench scenario filter')
    parser.add_argument('--rcabench_scenarios_file',
                       type=str,
                       default=None,
                       help='Optional text/json file containing RCAbench scenario names, one per line or as a JSON list')
    parser.add_argument('--failure_types', nargs='+',
                       default=['cpu_contention', 'network_delay', 'code_delay'],
                       help='Failure types to evaluate')

    parser.add_argument('--tau_max', type=int, default=3,
                       help='Maximum lag used by PCMCI')
    parser.add_argument('--top_k_services', type=int, default=10,
                       help='Number of Stage 1 candidate services')
    parser.add_argument('--top_k_pods', type=int, default=10,
                       help='Number of final pod candidates')
    parser.add_argument('--stage1_expand_factor', type=float, default=2.0,
                       help='Expansion factor used in Stage 1 candidate recall')
    parser.add_argument('--stage1_gnn_weight', type=float, default=0.45,
                       help='Weight of the GNN signal in Stage 1')
    parser.add_argument('--stage1_anomaly_weight', type=float, default=0.55,
                       help='Weight of the anomaly signal in Stage 1')
    parser.add_argument('--stage1_vote_weight', type=float, default=0.45,
                       help='Vote component weight in Stage 1 aggregation')
    parser.add_argument('--stage1_mean_weight', type=float, default=0.35,
                       help='Mean component weight in Stage 1 aggregation')
    parser.add_argument('--stage1_peak_weight', type=float, default=0.20,
                       help='Peak component weight in Stage 1 aggregation')
    parser.add_argument('--stage1_response_weight', type=float, default=0.50,
                       help='Extra Stage 1 weight for response-style failures')
    parser.add_argument('--stage1_dual_hit_bonus', type=float, default=0.15,
                       help='Bonus for candidates supported by both Stage 1 channels')
    parser.add_argument('--temporal_response_channel_enabled',
                       dest='temporal_response_channel_enabled',
                       action='store_true',
                       help='Enable the Stage 1 temporal response channel')
    parser.add_argument('--disable_temporal_response_channel',
                       dest='temporal_response_channel_enabled',
                       action='store_false',
                       help='Disable the Stage 1 temporal response channel for ablation')
    parser.add_argument('--dual_channel_consistency_enabled',
                       dest='dual_channel_consistency_enabled',
                       action='store_true',
                       help='Enable the Stage 1 dual-channel consistency reward')
    parser.add_argument('--disable_dual_channel_consistency',
                       dest='dual_channel_consistency_enabled',
                       action='store_false',
                       help='Disable the Stage 1 dual-channel consistency reward for ablation')
    parser.add_argument('--stage1_weak_type_conditional_expansion_enabled',
                       dest='stage1_weak_type_conditional_expansion_enabled',
                       action='store_true',
                       help='Expand Stage 1 recall only for configured weak failure types')
    parser.add_argument('--disable_stage1_weak_type_conditional_expansion',
                       dest='stage1_weak_type_conditional_expansion_enabled',
                       action='store_false',
                       help='Disable weak-type-only Stage 1 expansion')
    parser.add_argument('--stage1_weak_type_expand_factor', type=float, default=2.0,
                       help='Stage 1 expansion factor used only for weak failure types when enabled')
    parser.add_argument('--stage1_weak_failure_types', type=str,
                       default='stress,container-kill,pod-failure,exception,corrupt',
                       help='Comma-separated failure types that receive weak-type conditional Stage 1 expansion')
    parser.add_argument('--lag_prior_enabled', dest='lag_prior_enabled', action='store_true',
                       help='Enable lag-causal prior during Stage 1 encoding and prior-aware pruning')
    parser.add_argument('--disable_lag_prior', dest='lag_prior_enabled', action='store_false',
                       help='Disable lag-causal prior for ablation')
    parser.add_argument('--lag_alignment_enabled', dest='lag_alignment_enabled', action='store_true',
                       help='Enable Stage 2 dominant-lag alignment')
    parser.add_argument('--disable_lag_alignment', dest='lag_alignment_enabled', action='store_false',
                       help='Disable Stage 2 dominant-lag alignment for ablation')
    parser.add_argument('--counterfactual_effect_enabled', dest='counterfactual_effect_enabled', action='store_true',
                       help='Enable the Stage 2 counterfactual effect term')
    parser.add_argument('--disable_counterfactual_effect', dest='counterfactual_effect_enabled', action='store_false',
                       help='Disable the Stage 2 counterfactual effect term for ablation')
    parser.add_argument('--local_anomaly_score_enabled', dest='local_anomaly_score_enabled', action='store_true',
                       help='Enable the Stage 2 local anomaly/model score term')
    parser.add_argument('--disable_local_anomaly_score', dest='local_anomaly_score_enabled', action='store_false',
                       help='Disable the Stage 2 local anomaly/model score term for ablation')
    parser.add_argument('--stage2_stage1_weight', type=float, default=0.35,
                       help='Stage 1 contribution weight inside Stage 2 fusion')
    parser.add_argument('--stage2_stage2_weight', type=float, default=0.65,
                       help='Stage 2 score weight inside Stage 2 fusion')
    parser.add_argument('--stage2_keep_top1_anchor', dest='stage2_keep_top1_anchor', action='store_true',
                       help='Keep the current Stage 2 top-1 anchor before later reranking')
    parser.add_argument('--disable_stage2_keep_top1_anchor', dest='stage2_keep_top1_anchor', action='store_false',
                       help='Disable Stage 2 top-1 anchoring')
    parser.add_argument('--stage2_score_transform', type=str, default='log1p_clip',
                       choices=['none', 'clip', 'log1p', 'log1p_clip'],
                       help='Stage 2 raw score transform before reranking')
    parser.add_argument('--stage2_score_clip_percentile', type=float, default=95.0,
                       help='Percentile used by Stage 2 robust clipping')
    parser.add_argument('--stage2_score_clip_min_candidates', type=int, default=8,
                       help='Minimum candidate count before enabling percentile clipping')
    parser.add_argument('--stage2_shared_candidate_penalty', type=float, default=0.12,
                       help='Penalty applied to over-shared candidates during the final type-aware rerank')
    parser.add_argument('--type_aware_rerank_enabled', dest='type_aware_rerank_enabled', action='store_true',
                       help='Enable the final failure-type-aware rerank')
    parser.add_argument('--disable_type_aware_rerank', dest='type_aware_rerank_enabled', action='store_false',
                       help='Disable the final failure-type-aware rerank for clean ablations')
    parser.add_argument('--source_candidate_pool', type=int, default=24,
                       help='Candidate pool size for source-ness scoring')
    parser.add_argument('--source_score_weight', type=float, default=0.35,
                       help='Weight of the source-ness score in final fusion')
    parser.add_argument('--source_self_weight', type=float, default=0.40,
                       help='Weight of the self-advantage term in source-ness scoring')
    parser.add_argument('--source_delta_weight', type=float, default=0.25,
                       help='Weight of the delta-advantage term in source-ness scoring')
    parser.add_argument('--source_margin_weight', type=float, default=0.35,
                       help='Weight of the downstream-margin term in source-ness scoring')
    parser.add_argument('--enable_stage2_reranker', dest='enable_stage2_reranker', action='store_true',
                       help='Enable the Stage 2 leave-one-out reranker')
    parser.add_argument('--disable_stage2_reranker', dest='enable_stage2_reranker', action='store_false',
                       help='Disable the Stage 2 leave-one-out reranker')
    parser.add_argument('--reranker_candidate_pool', type=int, default=24,
                       help='Candidate pool size used by the Stage 2 reranker')
    parser.add_argument('--reranker_blend', type=float, default=0.75,
                       help='Blend ratio used by the Stage 2 reranker')
    parser.add_argument('--reranker_blend_replace_body', type=float, default=None,
                       help='Optional LGBM blend for replace-body only (default in code: 0.35 if omitted)')
    parser.add_argument('--reranker_epochs', type=int, default=200,
                       help='Training epochs used by the Stage 2 reranker')
    parser.add_argument('--reranker_lr', type=float, default=0.05,
                       help='Learning rate used by the Stage 2 reranker')
    parser.add_argument('--reranker_l2', type=float, default=1e-4,
                       help='L2 regularization used by the Stage 2 reranker')
    parser.add_argument('--stage2_reranker_dump', action='store_true',
                       help='Dump Stage 2 reranker training rows (features + label) to output_dir')
    parser.add_argument('--stage2_reranker_model_path', type=str, default=None,
                       help='Path to a trained Stage 2 reranker model (joblib)')
    parser.add_argument('--stage2_reranker_cv_predict', action='store_true',
                       help='Run leakage-free group K-fold OOF Stage 2 reranker and feed predictions back into RCA evaluation')
    parser.add_argument('--stage2_reranker_cv_folds', type=int, default=5,
                       help='Number of folds for --stage2_reranker_cv_predict')
    parser.add_argument('--stage2_reranker_cv_feature_set', type=str, default='hybrid',
                       choices=['base', 'enhanced', 'hybrid'],
                       help='Feature set used by --stage2_reranker_cv_predict')
    parser.add_argument('--stage2_reranker_cv_output_dir', type=str, default=None,
                       help='Optional output directory for OOF reranker artifacts')
    parser.add_argument('--code_top1_challenger_enabled', dest='code_top1_challenger_enabled', action='store_true',
                       help='Enable a lightweight top-1 challenger for replace-code/method/body faults')
    parser.add_argument('--disable_code_top1_challenger', dest='code_top1_challenger_enabled', action='store_false',
                       help='Disable the lightweight code-family top-1 challenger')
    parser.add_argument('--code_top1_challenger_topn', type=int, default=4,
                       help='How many leading candidates to inspect for the code-family top-1 challenger')
    parser.add_argument('--code_top1_challenger_margin', type=float, default=0.02,
                       help='Minimum challenger advantage required before promoting a code-family candidate to top-1')
    parser.add_argument('--global_hot_challenger_enabled', dest='global_hot_challenger_enabled', action='store_true',
                       help='Enable generic top-1 debiasing for globally hot candidates')
    parser.add_argument('--disable_global_hot_challenger', dest='global_hot_challenger_enabled', action='store_false',
                       help='Disable generic top-1 debiasing for globally hot candidates')
    parser.add_argument('--global_hot_challenger_topn', type=int, default=5,
                       help='How many leading candidates to inspect for the global-hot top-1 challenger')
    parser.add_argument('--global_hot_challenger_threshold', type=float, default=0.95,
                       help='Shared-candidate score threshold used to identify a globally hot incumbent')
    parser.add_argument('--global_hot_challenger_margin', type=float, default=0.30,
                       help='Maximum final-score gap allowed before a local challenger can replace a hot incumbent')
    parser.add_argument('--global_hot_challenger_evidence_margin', type=float, default=-0.05,
                       help='Minimum local-evidence advantage for a challenger after hot-candidate debiasing')
    parser.add_argument('--stage1_guard_enabled', dest='stage1_guard_enabled', action='store_true',
                       help='Enable a conservative guardrail that protects strong per-sample Stage 1 evidence in final reranking')
    parser.add_argument('--disable_stage1_guard', dest='stage1_guard_enabled', action='store_false',
                       help='Disable the Stage 1 strong-evidence guardrail')
    parser.add_argument('--stage1_guard_topn', type=int, default=3,
                       help='How many Stage 1 leading candidates can receive the guardrail bonus')
    parser.add_argument('--stage1_guard_bonus', type=float, default=0.06,
                       help='Maximum bonus applied by the Stage 1 strong-evidence guardrail')
    parser.add_argument('--stage1_guard_min_norm', type=float, default=0.75,
                       help='Minimum normalized Stage 1 score required to receive the guardrail bonus')
    parser.add_argument('--stage1_guard_gap', type=float, default=0.08,
                       help='Minimum normalized Stage 1 gap over the guard boundary before enabling the guardrail')
    parser.add_argument('--stage1_guard_preserve_top1', dest='stage1_guard_preserve_top1', action='store_true',
                       help='Cap the Stage 1 guardrail so it cannot replace the current top-1 candidate')
    parser.add_argument('--disable_stage1_guard_preserve_top1', dest='stage1_guard_preserve_top1', action='store_false',
                       help='Allow the Stage 1 guardrail to replace the current top-1 candidate')
    parser.add_argument('--dynamic_confidence_fusion_enabled', dest='dynamic_confidence_fusion_enabled', action='store_true',
                       help='Enable per-sample confidence-based dynamic Stage1/Stage2/source fusion weights')
    parser.add_argument('--disable_dynamic_confidence_fusion', dest='dynamic_confidence_fusion_enabled', action='store_false',
                       help='Disable per-sample confidence-based dynamic fusion')
    parser.add_argument('--dynamic_confidence_strength', type=float, default=0.35,
                       help='Strength of confidence-based weight adjustment')
    parser.add_argument('--dynamic_confidence_gap_topn', type=int, default=3,
                       help='Use top-1 minus top-n normalized score as the confidence gap')
    parser.add_argument('--dynamic_confidence_max_shift', type=float, default=0.18,
                       help='Maximum absolute per-channel weight shift before re-normalization')
    parser.add_argument('--stage1_source_aware_score_enabled', dest='stage1_source_aware_score_enabled', action='store_true',
                       help='Enable source-aware Stage1 score shaping for weak fault mechanisms')
    parser.add_argument('--disable_stage1_source_aware_score', dest='stage1_source_aware_score_enabled', action='store_false',
                       help='Disable source-aware Stage1 score shaping')
    parser.add_argument('--stage1_source_aware_strength', type=float, default=1.0,
                       help='Global multiplier for source-aware Stage1 score shaping weights')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Execution device, e.g. cpu or cuda')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Optional config file path')
    parser.add_argument('--save_config', action='store_true',
                       help='Save the resolved config to the output directory')

    parser.set_defaults(
        stage2_keep_top1_anchor=True,
        enable_stage2_reranker=True,
        code_top1_challenger_enabled=True,
        global_hot_challenger_enabled=False,
        stage1_guard_enabled=False,
        stage1_guard_preserve_top1=True,
        dynamic_confidence_fusion_enabled=False,
        stage1_source_aware_score_enabled=False,
        stage1_weak_type_conditional_expansion_enabled=False,
        temporal_response_channel_enabled=True,
        dual_channel_consistency_enabled=True,
        lag_prior_enabled=True,
        lag_alignment_enabled=True,
        counterfactual_effect_enabled=True,
        local_anomaly_score_enabled=True,
        type_aware_rerank_enabled=None,
    )
    return parser.parse_args()


def _load_scenarios_file(path: str) -> list:
    scenario_path = Path(path)
    if not scenario_path.is_absolute():
        scenario_path = REPO_ROOT / scenario_path
    if not scenario_path.exists():
        raise FileNotFoundError(f'RCAbench scenarios file not found: {scenario_path}')
    text = scenario_path.read_text(encoding='utf-8').strip()
    if not text:
        return []
    if scenario_path.suffix.lower() == '.json':
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ('scenarios', 'test_scenarios', 'train_scenarios'):
                if key in payload:
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError(f'RCAbench scenarios JSON must contain a list: {scenario_path}')
        return [str(x).strip() for x in payload if str(x).strip()]
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith('#')]

def load_config_from_args(args) -> LADSConfig:
    """从命令行参数加载配置"""
    if args.config_file and os.path.exists(args.config_file):
        logger.info(f'从文件加载配置: {args.config_file}')
        config = LADSConfig.load(args.config_file)
    else:
        config = LADSConfig()
    
    # 根据数据集类型设置默认 data_dir / output_dir，将"数据预处理输出"和"模型输出"分开
    if args.data_dir:
        data_dir = args.data_dir
        # 如果是相对路径，转换为绝对路径（相对于项目根目录）
        if not os.path.isabs(data_dir):
            data_dir = str(REPO_ROOT / data_dir)
    else:
        if getattr(args, 'dataset_type', 'trainticket') == 'aiops2025':
            # AIOps2025: 预处理数据全部在仓库根目录下的 output 中
            data_dir = str(REPO_ROOT / "output")
        elif getattr(args, 'dataset_type', 'trainticket') == 'rcabench':
            # RCAbench: 预处理数据建议放在仓库根目录下 RCAbench/rcabench_preprocessed
            data_dir = str(REPO_ROOT / "RCAbench" / "rcabench_preprocessed")
        else:
            # TrainTicket: 使用原有 graph_1 目录
            data_dir = str(REPO_ROOT / "data" / "TrainTicket_2024" / "graph_1")
    
    config.data.data_dir = data_dir
    config.data.train_dates = args.train_dates
    config.data.test_dates = args.test_dates
    config.data.failure_types = args.failure_types
    config.rca.tau_max = args.tau_max
    config.rca.top_k_services = args.top_k_services
    config.rca.top_k_pods = args.top_k_pods
    config.rca.stage1_expand_factor = args.stage1_expand_factor
    config.rca.stage1_gnn_weight = args.stage1_gnn_weight
    config.rca.stage1_anomaly_weight = args.stage1_anomaly_weight
    config.rca.stage1_vote_weight = args.stage1_vote_weight
    config.rca.stage1_mean_weight = args.stage1_mean_weight
    config.rca.stage1_peak_weight = args.stage1_peak_weight
    config.rca.stage1_response_weight = args.stage1_response_weight
    config.rca.stage1_dual_hit_bonus = args.stage1_dual_hit_bonus
    config.rca.temporal_response_channel_enabled = args.temporal_response_channel_enabled
    config.rca.dual_channel_consistency_enabled = args.dual_channel_consistency_enabled
    config.rca.stage1_weak_type_conditional_expansion_enabled = args.stage1_weak_type_conditional_expansion_enabled
    config.rca.stage1_weak_type_expand_factor = args.stage1_weak_type_expand_factor
    config.rca.stage1_weak_failure_types = args.stage1_weak_failure_types
    config.rca.lag_prior_enabled = args.lag_prior_enabled
    config.rca.lag_alignment_enabled = args.lag_alignment_enabled
    config.rca.counterfactual_effect_enabled = args.counterfactual_effect_enabled
    config.rca.local_anomaly_score_enabled = args.local_anomaly_score_enabled
    config.rca.stage2_stage1_weight = args.stage2_stage1_weight
    config.rca.stage2_stage2_weight = args.stage2_stage2_weight
    config.rca.stage2_keep_top1_anchor = args.stage2_keep_top1_anchor
    config.rca.stage2_score_transform = args.stage2_score_transform
    config.rca.stage2_score_clip_percentile = args.stage2_score_clip_percentile
    config.rca.stage2_score_clip_min_candidates = args.stage2_score_clip_min_candidates
    config.rca.stage2_shared_candidate_penalty = args.stage2_shared_candidate_penalty
    if args.type_aware_rerank_enabled is None:
        # The type-aware reranker was designed around RCAbench failure families.
        # Keep it on for RCAbench, but avoid applying those priors to AIOps2025.
        config.rca.type_aware_rerank_enabled = (
            getattr(args, 'dataset_type', 'trainticket') == 'rcabench'
        )
    else:
        config.rca.type_aware_rerank_enabled = args.type_aware_rerank_enabled
    config.rca.source_candidate_pool = args.source_candidate_pool
    config.rca.source_score_weight = args.source_score_weight
    config.rca.source_self_weight = args.source_self_weight
    config.rca.source_delta_weight = args.source_delta_weight
    config.rca.source_margin_weight = args.source_margin_weight
    config.rca.enable_stage2_reranker = args.enable_stage2_reranker
    config.rca.reranker_candidate_pool = args.reranker_candidate_pool
    config.rca.reranker_blend = args.reranker_blend
    config.rca.reranker_blend_replace_body = getattr(args, 'reranker_blend_replace_body', None)
    config.rca.reranker_epochs = args.reranker_epochs
    config.rca.reranker_lr = args.reranker_lr
    config.rca.reranker_l2 = args.reranker_l2
    config.rca.code_top1_challenger_enabled = args.code_top1_challenger_enabled
    config.rca.code_top1_challenger_topn = args.code_top1_challenger_topn
    config.rca.code_top1_challenger_margin = args.code_top1_challenger_margin
    config.rca.global_hot_challenger_enabled = args.global_hot_challenger_enabled
    config.rca.global_hot_challenger_topn = args.global_hot_challenger_topn
    config.rca.global_hot_challenger_threshold = args.global_hot_challenger_threshold
    config.rca.global_hot_challenger_margin = args.global_hot_challenger_margin
    config.rca.global_hot_challenger_evidence_margin = args.global_hot_challenger_evidence_margin
    config.rca.stage1_guard_enabled = args.stage1_guard_enabled
    config.rca.stage1_guard_topn = args.stage1_guard_topn
    config.rca.stage1_guard_bonus = args.stage1_guard_bonus
    config.rca.stage1_guard_min_norm = args.stage1_guard_min_norm
    config.rca.stage1_guard_gap = args.stage1_guard_gap
    config.rca.stage1_guard_preserve_top1 = args.stage1_guard_preserve_top1
    config.rca.dynamic_confidence_fusion_enabled = args.dynamic_confidence_fusion_enabled
    config.rca.dynamic_confidence_strength = args.dynamic_confidence_strength
    config.rca.dynamic_confidence_gap_topn = args.dynamic_confidence_gap_topn
    config.rca.dynamic_confidence_max_shift = args.dynamic_confidence_max_shift
    config.rca.stage1_source_aware_score_enabled = args.stage1_source_aware_score_enabled
    config.rca.stage1_source_aware_strength = args.stage1_source_aware_strength
    config.training.device = args.device
    config.seed = args.seed

    # 输出目录：与数据分离
    if args.output_dir:
        output_dir = args.output_dir
    else:
        if getattr(args, 'dataset_type', 'trainticket') == 'aiops2025':
            # AIOps2025 的 RCA / 模型输出放在 model_output/aiops2025 下
            output_dir = str(REPO_ROOT / "model_output" / "aiops2025")
        elif getattr(args, 'dataset_type', 'trainticket') == 'rcabench':
            output_dir = str(REPO_ROOT / "model_output" / "rcabench")
        else:
            # TrainTicket 默认 outputs
            output_dir = str(REPO_ROOT / "outputs" / "trainticket")
    config.output_dir = output_dir
    
    return config


def _normalize_groundtruth_entries(groundtruth) -> list:
    normalized = []
    if not groundtruth:
        return normalized
    try:
        items = list(groundtruth)
    except Exception:
        items = [groundtruth]
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            normalized.append({'ntype': str(item[0]), 'id': item[1]})
        else:
            normalized.append({'ntype': 'unknown', 'id': item})
    return normalized


def _extract_groundtruth_pod_ids(groundtruth) -> list:
    pod_ids = []
    for item in _normalize_groundtruth_entries(groundtruth):
        if item.get('ntype') == 'pod':
            pod_ids.append(item.get('id'))
    return pod_ids


def _build_rerank_debug_export(rca_results: dict, labels: list, groundtruths: list) -> dict:
    per_sample = (rca_results.get('per_sample_results') or []) if isinstance(rca_results, dict) else []
    shared_diag = (((rca_results.get('stage2') or {}).get('shared_diagnostics') or {}) if isinstance(rca_results, dict) else {})
    export = {
        'summary': {
            'n_samples': len(labels),
            'shared_top_candidate_pool': shared_diag.get('top_candidate_pool', []),
            'shared_top_ranking': shared_diag.get('top_shared_ranking', []),
            'stage2_transform': shared_diag.get('stage2_transform', {}),
            'shared_top_breakdown': shared_diag.get('top_shared_breakdown', []),
        },
        'samples': [],
    }

    for idx, label in enumerate(labels):
        per_sample_rec = per_sample[idx] if idx < len(per_sample) and isinstance(per_sample[idx], dict) else {}
        final_ranking = per_sample_rec.get('final_ranking', []) or []
        debug_candidates = per_sample_rec.get('debug_candidates', []) or []
        debug_weights = per_sample_rec.get('debug_weights', {}) or {}
        groundtruth = groundtruths[idx] if idx < len(groundtruths) else set()
        groundtruth_entries = _normalize_groundtruth_entries(groundtruth)
        groundtruth_pod_ids = _extract_groundtruth_pod_ids(groundtruth)
        final_rank_map = {}
        for rank, item in enumerate(final_ranking, start=1):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                final_rank_map[item[0]] = rank
            elif isinstance(item, dict):
                final_rank_map[item.get('id', item.get('pod_id'))] = rank

        debug_rank_map = {item.get('pod_id'): rank for rank, item in enumerate(debug_candidates, start=1) if isinstance(item, dict)}
        export['samples'].append({
            'sample_idx': idx,
            'timestamp': label.get('timestamp'),
            'failure_type': label.get('failure_type'),
            'cmdb_id': label.get('cmdb_id'),
            'groundtruth': groundtruth_entries,
            'groundtruth_pod_ids': groundtruth_pod_ids,
            'groundtruth_in_debug_candidates': any(pid in debug_rank_map for pid in groundtruth_pod_ids),
            'groundtruth_in_final_topk': any(pid in final_rank_map for pid in groundtruth_pod_ids),
            'groundtruth_debug_ranks': {str(pid): debug_rank_map[pid] for pid in groundtruth_pod_ids if pid in debug_rank_map},
            'groundtruth_final_ranks': {str(pid): final_rank_map[pid] for pid in groundtruth_pod_ids if pid in final_rank_map},
            'debug_weights': debug_weights,
            'final_ranking': final_ranking,
            'debug_candidates': debug_candidates,
        })
    return export


def _apply_oof_reranker_results_to_rca(rca_results: dict, oof_rankings_path: str) -> None:
    """Replace per-sample final rankings with leakage-free OOF reranker rankings."""
    import csv
    from collections import defaultdict

    rankings_by_sample = defaultdict(list)
    with open(oof_rankings_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sample_idx = int(row['sample_idx'])
                rank = int(row.get('rank', 0))
                pod_id = int(row['pod_id'])
                score = float(row.get('pred_score', 0.0))
            except Exception:
                continue
            rankings_by_sample[sample_idx].append((rank, pod_id, score))

    if not rankings_by_sample:
        raise ValueError(f'No OOF rankings found in {oof_rankings_path}')

    per_sample = rca_results.get('per_sample_results')
    if not isinstance(per_sample, list):
        per_sample = []
        rca_results['per_sample_results'] = per_sample

    max_idx = max(rankings_by_sample)
    while len(per_sample) <= max_idx:
        per_sample.append({})

    for sample_idx, rows in rankings_by_sample.items():
        rows = sorted(rows, key=lambda item: (item[0], item[1]))
        per_sample[sample_idx]['final_ranking'] = [(int(pid), float(score)) for _, pid, score in rows]
        per_sample[sample_idx]['oof_reranker_applied'] = True

    first_rows = sorted(rankings_by_sample.get(0, []), key=lambda item: (item[0], item[1]))
    if first_rows:
        rca_results['final_ranking'] = [(int(pid), float(score)) for _, pid, score in first_rows]
        if isinstance(rca_results.get('stage2'), dict):
            rca_results['stage2']['candidate_pods'] = rca_results['final_ranking']
            reranker_diag = rca_results['stage2'].setdefault('reranker', {})
            reranker_diag.update(
                {
                    'enabled': True,
                    'mode': 'oof_cv',
                    'oof_rankings_path': oof_rankings_path,
                    'per_sample_resync': 'oof_cv_rankings',
                }
            )


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 加载配置
    config = load_config_from_args(args)

    # 统一固定随机性（必须在模型初始化前完成）
    set_all_seeds(config.seed)
    logger.info(f'使用随机种子 seed={config.seed}')
    
    # 保存配置（如果指定）
    if args.save_config:
        os.makedirs(config.output_dir, exist_ok=True)
        config_path = os.path.join(config.output_dir, 'config.json')
        config.save(config_path)
        logger.info(f'配置已保存到: {config_path}')
    
    logger.info('=' * 80)
    logger.info('LADS-Causal: 时滞感知双层因果根因定位框架')
    logger.info('=' * 80)
    logger.info(f'输出目录: {config.output_dir}')
    logger.info(f'设备: {config.training.device}')
    logger.info(f'时滞范围: 0-{config.rca.tau_max}')
    
    # ===== 步骤1: 数据预处理 =====
    logger.info('\n' + '=' * 80)
    logger.info('步骤1: 数据预处理')
    logger.info('=' * 80)
    
    # 根据数据集类型选择不同的预处理器
    if getattr(args, 'dataset_type', 'trainticket') == 'aiops2025':
        logger.info('使用 AIOps2025DataPreprocessor')
        preprocessor = AIOps2025DataPreprocessor(config)
    elif getattr(args, 'dataset_type', 'trainticket') == 'rcabench':
        logger.info('使用 RCAbenchDataPreprocessor')
        preprocessor = RCAbenchDataPreprocessor(config)
    else:
        logger.info('使用 TrainTicketDataPreprocessor')
        preprocessor = TrainTicketDataPreprocessor(config)
    
    # 加载训练 / 测试数据集
    if getattr(args, 'dataset_type', 'trainticket') == 'aiops2025':
        # AIOps2025 当前只做测试评估，不需要训练集
        logger.info('AIOps2025: 跳过训练集加载，仅加载测试集')
        train_dataset = None
        test_dataset = preprocessor.load_test_dataset()
        test_dataset = preprocessor.scale_dataset(test_dataset, scalers_dir=config.output_dir)
    elif getattr(args, 'dataset_type', 'trainticket') == 'rcabench':
        # RCAbench：不使用训练集
        logger.info('RCAbench: 跳过训练集加载，仅加载测试集')
        train_dataset = None
        # scenario 列表：
        # - 若用户显式提供 --rcabench_scenarios，则按列表过滤
        # - 否则默认加载全部 scenarios（不要使用 TrainTicket 的默认 test_dates）
        if getattr(args, 'rcabench_scenarios_file', None):
            config.data.test_dates = _load_scenarios_file(args.rcabench_scenarios_file)
        elif getattr(args, 'rcabench_scenarios', None):
            config.data.test_dates = list(args.rcabench_scenarios)
        else:
            config.data.test_dates = []
        test_dataset = preprocessor.load_test_dataset()
        test_dataset = preprocessor.scale_dataset(test_dataset, scalers_dir=config.output_dir)
    else:
        # TrainTicket: 按原逻辑加载训练 + 测试
        train_dataset = preprocessor.load_train_dataset()
        train_dataset = preprocessor.scale_dataset(train_dataset)
        test_dataset = preprocessor.load_test_dataset()
        test_dataset = preprocessor.scale_dataset(test_dataset, scalers_dir=config.output_dir)
    
    # 准备RCA数据
    rca_data = preprocessor.prepare_data_for_rca(test_dataset)
    
    graphs = rca_data['graphs']
    stacked_nfeat = rca_data['stacked_nfeat']
    labels = rca_data['labels']
    groundtruths = rca_data['groundtruths']
    nan_nodes = rca_data['nan_nodes']
    stage1_channel_aux = rca_data.get('stage1_channel_aux')
    
    if getattr(args, 'dataset_type', 'trainticket') == 'aiops2025' and labels:
        actual_types = sorted(set(str(l.get('failure_type', '')).strip() for l in labels if l.get('failure_type')))
        if actual_types:
            config.data.failure_types = actual_types
            logger.info(f'AIOps2025 按实际故障类型评估: {config.data.failure_types}')
    if getattr(args, 'dataset_type', 'trainticket') == 'rcabench' and labels:
        actual_types = sorted(set(str(l.get('failure_type', '')).strip() for l in labels if l.get('failure_type')))
        if actual_types:
            config.data.failure_types = actual_types
            logger.info(f'RCAbench 按实际故障类型评估: {config.data.failure_types}')
    
    # 提取服务级时间序列（如果存在，用于AIOps2025）
    service_timeseries = rca_data.get('service_timeseries')
    service_list = rca_data.get('service_list')
    G_trace_edges = rca_data.get('G_trace_edges')
    X_norm = rca_data.get('X_norm')
    
    if service_timeseries is not None:
        logger.info(f'检测到服务级时间序列: shape={service_timeseries.shape}, '
                   f'服务数={len(service_list) if service_list else 0}, '
                   f'边数={len(G_trace_edges) if G_trace_edges else 0}')
    
    # 计算数据统计（简化：使用均值）
    # 实际应该使用训练好的模型进行预测
    data_stats = {'mean': {}, 'cov_inv': {}}
    for ntype in ['api', 'pod']:
        if ntype in stacked_nfeat:
            feats = stacked_nfeat[ntype]  # (T, N, F)
            mean_feat = th.mean(feats, dim=0)  # (N, F)
            data_stats['mean'][ntype] = mean_feat.to(config.training.device)
            
            # 简化：使用单位矩阵作为协方差逆矩阵
            num_nodes = feats.shape[1]
            num_feats = feats.shape[2]
            cov_inv = th.eye(num_feats).unsqueeze(0).repeat(num_nodes, 1, 1).to(config.training.device)
            data_stats['cov_inv'][ntype] = cov_inv
    
    # 如果使用服务级时间序列，也需要计算其统计信息
    if service_timeseries is not None:
        # 使用服务级时间序列的均值作为统计信息
        mean_feat = th.mean(service_timeseries, dim=0)  # (N, F)
        data_stats['mean']['service'] = mean_feat.to(config.training.device)
        
        num_nodes = service_timeseries.shape[1]
        num_feats = service_timeseries.shape[2]
        cov_inv = th.eye(num_feats).unsqueeze(0).repeat(num_nodes, 1, 1).to(config.training.device)
        data_stats['cov_inv']['service'] = cov_inv
    
    # Build Stage 2 node features. RCAbench has a dataset-specific helper; other
    # datasets can fall back to the stacked pod node features.
    if hasattr(preprocessor, 'build_pod_features'):
        pod_feats, norm_pod_feats, downstream_feats = preprocessor.build_pod_features(
            test_dataset, stacked_nfeat, config.rca.tau_max
        )
    else:
        logger.info('Preprocessor has no build_pod_features; using stacked pod features for Stage 2')
        pod_feats, norm_pod_feats, downstream_feats = {}, {}, {}
        pod_feat_seq = stacked_nfeat.get('pod') if isinstance(stacked_nfeat, dict) else None
        if pod_feat_seq is not None and pod_feat_seq.dim() == 3:
            pod_feat_seq_stage2 = pod_feat_seq.float().clone()
            if pod_feat_seq_stage2.shape[-1] > 2:
                metric_part = pod_feat_seq_stage2[:, :, 2:]
                metric_min = metric_part.min(dim=1, keepdim=True).values
                metric_max = metric_part.max(dim=1, keepdim=True).values
                metric_rng = metric_max - metric_min
                metric_rel = th.where(
                    metric_rng > 1e-6,
                    (metric_part - metric_min) / (metric_rng + 1e-6),
                    th.zeros_like(metric_part),
                )
                pod_feat_seq_stage2[:, :, 2:] = metric_rel
                logger.info('Stage 2 fallback uses event-local metric normalization for non-RCAbench data')

            num_pods = int(pod_feat_seq.shape[1])
            seq_len = int(pod_feat_seq.shape[0])
            for pod_id in range(num_pods):
                service_id = pod_id
                pod_mean = th.zeros_like(pod_feat_seq_stage2[0, pod_id, :])
                pod_mean = th.where(th.isnan(pod_mean), th.zeros_like(pod_mean), pod_mean)
                norm_pod_feats[service_id] = pod_mean

                hist_feats = {}
                for lag in range(config.rca.tau_max + 1):
                    src_idx = max(seq_len - 1 - lag, 0)
                    feat_lag = pod_feat_seq_stage2[src_idx, pod_id, :]
                    feat_lag = th.where(th.isnan(feat_lag), th.zeros_like(feat_lag), feat_lag)
                    hist_feats[lag] = feat_lag

                pod_feats[service_id] = {pod_id: hist_feats}
                downstream_feats[service_id] = hist_feats.get(0, pod_mean)

            logger.info(
                f'Stage 2 fallback features built: {len(pod_feats)} services, '
                f'{sum(len(pods) for pods in pod_feats.values())} pods'
            )
        else:
            logger.warning('No usable pod stacked features for Stage 2 fallback')
    
    logger.info(f'数据准备完成: {len(graphs)} 个图, {len(labels)} 个标签')
    
    # ===== 步骤2: 运行LADS-Causal框架 =====
    logger.info('\n' + '=' * 80)
    logger.info('步骤2: 运行LADS-Causal框架')
    logger.info('=' * 80)
    
    # 运行双层RCA（传入所有图）
    if not graphs:
        logger.error('没有可用的图数据')
        return
    
    stage2_reranker_dump_path = (
        os.path.join(config.output_dir, 'stage2_reranker_train.csv')
        if getattr(args, 'stage2_reranker_dump', False) or getattr(args, 'stage2_reranker_cv_predict', False)
        else None
    )

    rca_results = lag_aware_dual_stage_rca(
        graphs=graphs,
        stacked_nfeat=stacked_nfeat,
        data_stats=data_stats,
        labels=labels,
        device=config.training.device,
        tau_max=config.rca.tau_max,
        top_k_services=config.rca.top_k_services,
        top_k_pods=config.rca.top_k_pods,
        service_timeseries=service_timeseries,  # 新增：服务级时间序列
        service_list=service_list,  # 新增：服务列表
        G_trace_edges=G_trace_edges,  # 新增：Trace拓扑边列表
        nan_nodes=nan_nodes,
        pod_feats=pod_feats,
        norm_pod_feats=norm_pod_feats,
        downstream_feats=downstream_feats,
        stage1_channel_aux=stage1_channel_aux,
        groundtruths=groundtruths,
        stage2_reranker_dump_path=stage2_reranker_dump_path,
        stage2_reranker_model_path=getattr(args, 'stage2_reranker_model_path', None),
        rca_config=config.rca,
    )

    if getattr(args, 'stage2_reranker_cv_predict', False):
        if run_oof_cv_from_csv is None:
            raise ImportError(
                "stage2_reranker_split_eval.py is not included in the minimal RippleRCA release. "
                "Disable --stage2_reranker_cv_predict to run the main pipeline."
            )
        if not stage2_reranker_dump_path or not os.path.exists(stage2_reranker_dump_path):
            raise FileNotFoundError(f'OOF reranker dump CSV was not created: {stage2_reranker_dump_path}')
        oof_output_dir = (
            getattr(args, 'stage2_reranker_cv_output_dir', None)
            or os.path.join(config.output_dir, 'stage2_reranker_oof')
        )
        logger.info(
            f'Stage 2 OOF reranker: folds={args.stage2_reranker_cv_folds}, '
            f'feature_set={args.stage2_reranker_cv_feature_set}, output={oof_output_dir}'
        )
        run_oof_cv_from_csv(
            train_csv=stage2_reranker_dump_path,
            output_dir=oof_output_dir,
            seed=config.seed,
            cv_folds=int(args.stage2_reranker_cv_folds),
            feature_set=str(args.stage2_reranker_cv_feature_set),
            top_k=config.evaluation.compute_top_k,
        )
        oof_rankings_path = os.path.join(oof_output_dir, 'oof_rankings.csv')
        _apply_oof_reranker_results_to_rca(rca_results, oof_rankings_path)
        logger.info(f'Stage 2 OOF reranker rankings applied from: {oof_rankings_path}')
    
    # 保存RCA结果
    # 确保输出目录存在
    os.makedirs(config.output_dir, exist_ok=True)
    results_file = os.path.join(config.output_dir, 'rca_results.json')
    # 转换不可序列化的对象
    results_serializable = {
        'stage1': {
            'candidate_services': rca_results['stage1']['candidate_services'],
            'tau_star': {str(k): v for k, v in rca_results['stage1']['tau_star'].items()},
        },
        'stage2': {
            'candidate_pods': rca_results['stage2']['candidate_pods'],
        },
        'final_ranking': rca_results['final_ranking'],
        'per_sample_results': rca_results.get('per_sample_results', []),
    }
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    rerank_debug_file = os.path.join(config.output_dir, 'rerank_debug.json')
    rerank_debug = _build_rerank_debug_export(
        rca_results=rca_results,
        labels=labels,
        groundtruths=groundtruths,
    )
    with open(rerank_debug_file, 'w', encoding='utf-8') as f:
        json.dump(rerank_debug, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f'RCA结果已保存到: {results_file}')
    
    # ===== 步骤3: 评估结果 =====
    logger.info('\n' + '=' * 80)
    logger.info('步骤3: 评估结果')
    logger.info('=' * 80)
    
    # AIOps2025：若有按 groundtruth 每行的 labels/groundtruths，用其做评估，可得到多种故障类型及多样本
    eval_labels = labels
    eval_groundtruths = groundtruths
    per_row_graph_indices = None
    if getattr(args, 'dataset_type', 'trainticket') == 'aiops2025' and hasattr(test_dataset, 'per_row_labels') and hasattr(test_dataset, 'per_row_groundtruths'):
        if getattr(test_dataset, 'per_row_labels', None) and getattr(test_dataset, 'per_row_groundtruths', None):
            eval_labels = test_dataset.per_row_labels
            eval_groundtruths = test_dataset.per_row_groundtruths
            per_row_graph_indices = getattr(test_dataset, 'per_row_graph_indices', None)
            config.data.failure_types = sorted(set(str(l.get('failure_type', '')).strip() for l in eval_labels if l.get('failure_type')))
            logger.info(f'AIOps2025 按 groundtruth 行评估: {len(eval_labels)} 条, 故障类型: {config.data.failure_types}')

    # 若 RCA 阶段已按事件展开（graphs 数量与 eval_labels 对齐），但未提供 per_row_graph_indices，
    # 则构造 identity 映射以启用 per-sample 排序评估。
    if getattr(args, 'dataset_type', 'trainticket') == 'aiops2025' and per_row_graph_indices is None:
        try:
            if graphs and eval_labels and len(graphs) == len(eval_labels):
                per_row_graph_indices = list(range(len(eval_labels)))
        except Exception:
            pass

    # RCAbench：也使用 per-sample 的 final_ranking（rca_results 内会包含 per_sample_results）
    if getattr(args, 'dataset_type', 'trainticket') == 'rcabench' and per_row_graph_indices is None:
        try:
            per_sample = (rca_results.get('per_sample_results') if isinstance(rca_results, dict) else None)
            if per_sample and eval_labels and len(per_sample) == len(eval_labels):
                per_row_graph_indices = list(range(len(eval_labels)))
        except Exception:
            pass
    
    evaluation_results = evaluate_lads_results(
        rca_results=rca_results,
        groundtruths=eval_groundtruths,
        labels=eval_labels,
        failure_types=config.data.failure_types,
        top_k_list=config.evaluation.compute_top_k,
        per_row_graph_indices=per_row_graph_indices,
    )
    
    # 保存评估结果
    eval_file = os.path.join(config.output_dir, 'evaluation_results.json')
    # 转换numpy类型以便序列化
    eval_serializable = json.loads(json.dumps(evaluation_results, default=str))
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(eval_serializable, f, indent=2, ensure_ascii=False)
    logger.info(f'评估结果已保存到: {eval_file}')
    
       
    return rca_results, evaluation_results

if __name__ == '__main__':

    main()
