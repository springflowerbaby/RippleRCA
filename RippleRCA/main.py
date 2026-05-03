"""Public entrypoint for running RippleRCA experiments."""

from __future__ import annotations
import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List
import numpy as np  # pyright: ignore[reportMissingImports]
import torch as th  # pyright: ignore[reportMissingImports]
from config import RippleConfig
from data_preprocessor import AIOps2025DataPreprocessor, RCAbenchDataPreprocessor
from evaluator import evaluate_results
from lag_aware_rca import lag_aware_dual_stage_rca
from log import Logger  # pyright: ignore[reportMissingImports]

logger = Logger(__name__)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
    try:
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False
    except Exception:
        pass
    try:
        import dgl  # pyright: ignore[reportMissingImports]

        if hasattr(dgl, "seed"):
            dgl.seed(seed)
        elif hasattr(dgl, "random") and hasattr(dgl.random, "seed"):
            dgl.random.seed(seed)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RippleRCA: hierarchical lag-calibrated causal inference for microservice RCA"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["aiops2025", "rcabench"],
        default="rcabench",
        help="Dataset type.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--test_dates",
        nargs="+",
        default=None,
        help="Test dates for AIOps2025 or scenario names for RCAbench.",
    )
    parser.add_argument(
        "--failure_types",
        nargs="+",
        default=None,
        help="Optional failure-type filter.",
    )
    parser.add_argument(
        "--rcabench_scenarios",
        nargs="+",
        default=None,
        help="Optional RCAbench scenario list.",
    )
    parser.add_argument(
        "--rcabench_scenarios_file",
        type=str,
        default=None,
        help="Optional text or JSON file containing RCAbench scenario names.",
    )
    parser.add_argument("--tau_max", type=int, default=2, help="Maximum lag window.")
    parser.add_argument(
        "--top_k_services",
        type=int,
        default=8,
        help="Stage 1 candidate service count.",
    )
    parser.add_argument(
        "--top_k_pods",
        type=int,
        default=10,
        help="Final candidate count.",
    )
    parser.add_argument(
        "--stage2_stage1_weight",
        type=float,
        default=0.35,
        help="Stage 1 fusion weight in the final score.",
    )
    parser.add_argument(
        "--stage2_stage2_weight",
        type=float,
        default=0.65,
        help="Stage 2 fusion weight in the final score.",
    )
    parser.add_argument(
        "--temporal_response_channel_enabled",
        dest="temporal_response_channel_enabled",
        action="store_true",
        help="Enable the propagation response channel.",
    )
    parser.add_argument(
        "--disable_temporal_response_channel",
        dest="temporal_response_channel_enabled",
        action="store_false",
        help="Disable the propagation response channel.",
    )
    parser.add_argument(
        "--dual_channel_consistency_enabled",
        dest="dual_channel_consistency_enabled",
        action="store_true",
        help="Enable dual-channel consistency reward.",
    )
    parser.add_argument(
        "--disable_dual_channel_consistency",
        dest="dual_channel_consistency_enabled",
        action="store_false",
        help="Disable dual-channel consistency reward.",
    )
    parser.add_argument(
        "--lag_prior_enabled",
        dest="lag_prior_enabled",
        action="store_true",
        help="Enable topology-constrained lag priors.",
    )
    parser.add_argument(
        "--disable_lag_prior",
        dest="lag_prior_enabled",
        action="store_false",
        help="Disable topology-constrained lag priors.",
    )
    parser.add_argument(
        "--lag_alignment_enabled",
        dest="lag_alignment_enabled",
        action="store_true",
        help="Enable dominant-lag alignment in Stage 2.",
    )
    parser.add_argument(
        "--disable_lag_alignment",
        dest="lag_alignment_enabled",
        action="store_false",
        help="Disable dominant-lag alignment in Stage 2.",
    )
    parser.add_argument(
        "--counterfactual_effect_enabled",
        dest="counterfactual_effect_enabled",
        action="store_true",
        help="Enable the counterfactual effect term.",
    )
    parser.add_argument(
        "--disable_counterfactual_effect",
        dest="counterfactual_effect_enabled",
        action="store_false",
        help="Disable the counterfactual effect term.",
    )
    parser.add_argument(
        "--local_anomaly_score_enabled",
        dest="local_anomaly_score_enabled",
        action="store_true",
        help="Enable the local anomaly term.",
    )
    parser.add_argument(
        "--disable_local_anomaly_score",
        dest="local_anomaly_score_enabled",
        action="store_false",
        help="Disable the local anomaly term.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Execution device.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Optional JSON config file.",
    )
    parser.add_argument(
        "--save_config",
        action="store_true",
        help="Save the resolved config to the output directory.",
    )
    parser.set_defaults(
        temporal_response_channel_enabled=True,
        dual_channel_consistency_enabled=True,
        lag_prior_enabled=True,
        lag_alignment_enabled=True,
        counterfactual_effect_enabled=True,
        local_anomaly_score_enabled=True,
    )
    return parser.parse_args()


def _load_scenarios_file(path: str) -> List[str]:
    scenario_path = Path(path)
    if not scenario_path.exists():
        raise FileNotFoundError(f"RCAbench scenarios file not found: {scenario_path}")
    text = scenario_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if scenario_path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("scenarios", "test_scenarios", "train_scenarios"):
                if key in payload:
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError(
                f"RCAbench scenarios JSON must contain a list: {scenario_path}"
            )
        return [str(item).strip() for item in payload if str(item).strip()]
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def load_config_from_args(args: argparse.Namespace) -> RippleConfig:
    if args.config_file and os.path.exists(args.config_file):
        config = RippleConfig.load(args.config_file)
    else:
        config = RippleConfig()
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.test_dates is not None:
        config.data.test_dates = list(args.test_dates)
    if args.failure_types is not None:
        config.data.failure_types = list(args.failure_types)
    config.rca.tau_max = args.tau_max
    config.rca.top_k_services = args.top_k_services
    config.rca.top_k_pods = args.top_k_pods
    config.rca.stage2_stage1_weight = args.stage2_stage1_weight
    config.rca.stage2_stage2_weight = args.stage2_stage2_weight
    config.rca.temporal_response_channel_enabled = (
        args.temporal_response_channel_enabled
    )
    config.rca.dual_channel_consistency_enabled = args.dual_channel_consistency_enabled
    config.rca.lag_prior_enabled = args.lag_prior_enabled
    config.rca.lag_alignment_enabled = args.lag_alignment_enabled
    config.rca.counterfactual_effect_enabled = args.counterfactual_effect_enabled
    config.rca.local_anomaly_score_enabled = args.local_anomaly_score_enabled
    config.training.device = args.device
    config.output_dir = args.output_dir
    config.seed = args.seed
    return config


def _build_data_stats(
    stacked_nfeat: Dict[str, th.Tensor], device: str
) -> Dict[str, Dict[str, th.Tensor]]:
    data_stats: Dict[str, Dict[str, th.Tensor]] = {"mean": {}, "cov_inv": {}}
    for ntype, feats in stacked_nfeat.items():
        if feats is None or feats.numel() == 0:
            continue
        mean_feat = th.mean(feats, dim=0)
        data_stats["mean"][ntype] = mean_feat.to(device)
        num_nodes = feats.shape[1]
        num_feats = feats.shape[2]
        cov_inv = th.eye(num_feats).unsqueeze(0).repeat(num_nodes, 1, 1).to(device)
        data_stats["cov_inv"][ntype] = cov_inv
    return data_stats


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, th.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _save_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_to_serializable(payload), handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    config = load_config_from_args(args)
    set_all_seeds(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    if args.save_config:
        config.save(os.path.join(config.output_dir, "config.json"))
    if args.dataset_type == "aiops2025":
        preprocessor = AIOps2025DataPreprocessor(config)
    else:
        if args.rcabench_scenarios_file:
            config.data.test_dates = _load_scenarios_file(args.rcabench_scenarios_file)
        elif args.rcabench_scenarios:
            config.data.test_dates = list(args.rcabench_scenarios)
        preprocessor = RCAbenchDataPreprocessor(config)
    logger.info("Loading test dataset")
    test_dataset = preprocessor.load_test_dataset()
    test_dataset = preprocessor.scale_dataset(
        test_dataset, scalers_dir=config.output_dir
    )
    logger.info("Preparing RCA inputs")
    rca_data = preprocessor.prepare_data_for_rca(test_dataset)
    graphs = rca_data["graphs"]
    labels = rca_data["labels"]
    groundtruths = rca_data["groundtruths"]
    stacked_nfeat = rca_data["stacked_nfeat"]
    data_stats = _build_data_stats(stacked_nfeat, config.training.device)
    if not graphs:
        raise RuntimeError("No evaluation samples were loaded.")
    if labels:
        actual_types = sorted(
            {
                str(label.get("failure_type", "")).strip()
                for label in labels
                if str(label.get("failure_type", "")).strip()
            }
        )
        if actual_types:
            config.data.failure_types = actual_types
    service_timeseries = rca_data.get("service_timeseries")
    service_list = rca_data.get("service_list")
    trace_edges = rca_data.get("G_trace_edges")
    pod_feats: Dict[Any, Any] = {}
    norm_pod_feats: Dict[Any, Any] = {}
    downstream_feats: Dict[Any, Any] = {}
    if hasattr(preprocessor, "build_pod_features"):
        pod_feats, norm_pod_feats, downstream_feats = preprocessor.build_pod_features(
            test_dataset,
            stacked_nfeat,
            config.rca.tau_max,
        )
    logger.info("Running RippleRCA")
    rca_results = lag_aware_dual_stage_rca(
        graphs=graphs,
        stacked_nfeat=stacked_nfeat,
        data_stats=data_stats,
        labels=labels,
        device=config.training.device,
        tau_max=config.rca.tau_max,
        top_k_services=config.rca.top_k_services,
        top_k_pods=config.rca.top_k_pods,
        service_timeseries=service_timeseries,
        service_list=service_list,
        G_trace_edges=trace_edges,
        nan_nodes=rca_data.get("nan_nodes"),
        pod_feats=pod_feats,
        norm_pod_feats=norm_pod_feats,
        downstream_feats=downstream_feats,
        stage1_channel_aux=rca_data.get("stage1_channel_aux"),
        groundtruths=groundtruths,
        rca_config=config.rca,
    )
    per_row_graph_indices = rca_data.get("per_row_graph_indices")
    evaluation_results = evaluate_results(
        rca_results=rca_results,
        groundtruths=groundtruths,
        labels=labels,
        failure_types=config.data.failure_types,
        top_k_list=config.evaluation.compute_top_k,
        per_row_graph_indices=per_row_graph_indices,
    )
    results_path = os.path.join(config.output_dir, "rca_results.json")
    metrics_path = os.path.join(config.output_dir, "evaluation_results.json")
    _save_json(results_path, rca_results)
    _save_json(metrics_path, evaluation_results)
    logger.info("Finished RippleRCA evaluation")
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"Metrics saved to: {metrics_path}")
    overall = evaluation_results.get("overall", {})
    logger.info(
        "Overall metrics: "
        f"AC@1={overall.get('top-1', 0.0):.3f}, "
        f"Avg@5={overall.get('avg@5', 0.0):.3f}, "
        f"MRR={overall.get('mrr', 0.0):.3f}"
    )


if __name__ == "__main__":
    main()
