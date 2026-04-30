"""
配置管理模块

统一管理LADS-Causal框架的所有配置参数。
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = "./data"
    train_dates: List[str] = None
    test_dates: List[str] = None
    failure_types: List[str] = None
    failure_duration: int = 15
    
    # 特征选择
    nfeat_select: Dict[str, List[str]] = None
    
    # 数据处理
    node_scale_type: str = 'nodewise'
    edge_scale_type: str = 'edgewise'
    log_before_scale: bool = True
    process_miss: str = 'interpolate'
    process_extreme: bool = False
    k_sigma: int = 3
    
    def __post_init__(self):
        if self.train_dates is None:
            self.train_dates = ['2024-04-05', '2024-04-06']
        if self.test_dates is None:
            self.test_dates = ['2024-04-05', '2024-04-06']
        if self.failure_types is None:
            self.failure_types = []
        if self.nfeat_select is None:
            self.nfeat_select = {
                "api": ["count", "mean", "q60", "q70", "q80", "q90", "q95", "max"],
                "pod": [
                    "CpuUsage(m)", "CpuUsageRate(%)", "MemoryUsage(Mi)", "MemoryUsageRate(%)",
                    "SyscallRead", "SyscallWrite", "NetworkReceiveBytes", "NetworkTransmitBytes"
                ]
            }


@dataclass
class ModelConfig:
    """模型配置"""
    model_type: str = 'LagAwareGNN'
    hidden_feats: int = 64
    out_feats: int = 32
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.0


@dataclass
class RCAConfig:
    """RCA配置"""
    # Stage 1配置
    tau_max: int = 3
    tau_min: int = 0
    pcmci_alpha: float = 0.05
    top_k_services: int = 10
    stage1_expand_factor: float = 2.0
    stage1_gnn_weight: float = 0.45
    stage1_anomaly_weight: float = 0.55
    stage1_vote_weight: float = 0.45
    stage1_mean_weight: float = 0.35
    stage1_peak_weight: float = 0.20
    stage1_response_weight: float = 0.50
    stage1_dual_hit_bonus: float = 0.15
    temporal_response_channel_enabled: bool = True
    dual_channel_consistency_enabled: bool = True
    stage1_weak_type_conditional_expansion_enabled: bool = False
    stage1_weak_type_expand_factor: float = 2.0
    stage1_weak_failure_types: str = 'stress,container-kill,pod-failure,exception,corrupt'
    lag_prior_enabled: bool = True
    lag_alignment_enabled: bool = True
    counterfactual_effect_enabled: bool = True
    local_anomaly_score_enabled: bool = True
    
    # Stage 2配置
    top_k_pods: int = 10
    stage2_stage1_weight: float = 0.35
    stage2_stage2_weight: float = 0.65
    stage2_keep_top1_anchor: bool = True
    stage2_anchor_margin: float = 1e-6
    stage2_score_transform: str = 'log1p_clip'
    stage2_score_clip_percentile: float = 95.0
    stage2_score_clip_min_candidates: int = 8
    stage2_shared_candidate_penalty: float = 0.12
    type_aware_rerank_enabled: bool = True
    source_candidate_pool: int = 24
    source_score_weight: float = 0.35
    source_self_weight: float = 0.40
    source_delta_weight: float = 0.25
    source_margin_weight: float = 0.35
    enable_stage2_reranker: bool = True
    reranker_candidate_pool: int = 24
    reranker_blend: float = 0.75
    reranker_blend_replace_body: Optional[float] = None
    reranker_epochs: int = 200
    reranker_lr: float = 0.05
    reranker_l2: float = 1e-4
    code_top1_challenger_enabled: bool = True
    code_top1_challenger_topn: int = 4
    code_top1_challenger_margin: float = 0.02
    lambda_1: float = 0.6  # ModelScore权重
    lambda_2: float = 0.4  # CounterfactualEffect权重
    
    # 数据统计窗口
    window: int = 720  # 异常分数规范化窗口（分钟）
    score_type: str = 'pred_and_mean'
    dist_type: str = 'euclidean'
    
    # 阈值
    causal_threshold: float = 0.6
    ad_threshold: float = 3.0


@dataclass
class TrainingConfig:
    """训练配置"""
    device: str = 'cpu'
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 100
    valid_ratio: float = 0.0


@dataclass
class EvaluationConfig:
    """评估配置"""
    compute_top_k: List[int] = None
    aggregate_results: bool = False
    agg_duration: int = 2  # 聚合窗口（分钟）
    
    def __post_init__(self):
        if self.compute_top_k is None:
            self.compute_top_k = [1, 3, 5]


@dataclass
class LADSConfig:
    """LADS-Causal框架总配置"""
    data: DataConfig = None
    model: ModelConfig = None
    rca: RCAConfig = None
    training: TrainingConfig = None
    evaluation: EvaluationConfig = None
    
    # 路径配置
    output_dir: str = './outputs'
    cache_dir: str = './cache'
    log_dir: str = './logs'
    
    # 其他配置
    seed: int = 42
    debug: bool = False
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.rca is None:
            self.rca = RCAConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'data': self.data.__dict__ if self.data else {},
            'model': self.model.__dict__ if self.model else {},
            'rca': self.rca.__dict__ if self.rca else {},
            'training': self.training.__dict__ if self.training else {},
            'evaluation': self.evaluation.__dict__ if self.evaluation else {},
            'output_dir': self.output_dir,
            'cache_dir': self.cache_dir,
            'log_dir': self.log_dir,
            'seed': self.seed,
            'debug': self.debug,
        }
    
    def save(self, filepath: str):
        """保存配置到文件"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str):
        """从文件加载配置"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = cls()
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'rca' in config_dict:
            config.rca = RCAConfig(**config_dict['rca'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        if 'evaluation' in config_dict:
            config.evaluation = EvaluationConfig(**config_dict['evaluation'])
        
        config.output_dir = config_dict.get('output_dir', './outputs')
        config.cache_dir = config_dict.get('cache_dir', './cache')
        config.log_dir = config_dict.get('log_dir', './logs')
        config.seed = config_dict.get('seed', 42)
        config.debug = config_dict.get('debug', False)
        
        return config


# 默认配置实例
default_config = LADSConfig()
