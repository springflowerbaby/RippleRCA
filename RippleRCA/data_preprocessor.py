"""
数据预处理模块

处理TrainTicket数据，为LADS-Causal框架准备输入数据。
"""

import os
import torch as th  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

# 添加父目录到路径以便导入causelens模块

from dataset_trainticket_2024 import DatasetTrainTicket2024  # pyright: ignore[reportMissingImports]
from dataset_aiops_2025 import DatasetAIOps2025  # pyright: ignore[reportMissingImports]
from dataset_rcabench import DatasetRCAbench  # pyright: ignore[reportMissingImports]
from dataset import myDGLDataset  # pyright: ignore[reportMissingImports]
from log import Logger  # pyright: ignore[reportMissingImports]

# 导入数据流水线模块
from data_pipeline import TraceParser, MetricAggregator, build_service_timeseries  # pyright: ignore[reportMissingImports]

logger = Logger(__name__)


class TrainTicketDataPreprocessor:
    """TrainTicket数据预处理器"""
    
    def __init__(self, config):
        """
        Args:
            config: LADSConfig对象
        """
        self.config = config
        self.data_config = config.data
        self.rca_config = config.rca
        
        # 数据集实例
        self.train_dataset: Optional[myDGLDataset] = None
        self.test_dataset: Optional[myDGLDataset] = None
    
    def load_train_dataset(self, max_samples: int = 1e9) -> myDGLDataset:
        """加载训练数据集"""
        logger.info('加载训练数据集...')
        
        self.train_dataset = DatasetTrainTicket2024(
            data_dir=self.data_config.data_dir,
            dates=self.data_config.train_dates,
            node_feature_selector=self.data_config.nfeat_select,
            edge_reverse=False,
            add_self_loop=False,
            max_samples=max_samples,
            failure_types=[''],  # 训练时使用正常数据
            failure_duration=self.data_config.failure_duration,
            is_mask=False,
            process_miss=self.data_config.process_miss,
            process_extreme=self.data_config.process_extreme,
            k_sigma=self.data_config.k_sigma,
            use_split_info=False,
        )
        
        logger.info(f'训练数据集加载完成: {len(self.train_dataset)} 个样本')
        return self.train_dataset
    
    def load_test_dataset(self, max_samples: int = 1e9) -> myDGLDataset:
        """加载测试数据集"""
        logger.info('加载测试数据集...')
        
        self.test_dataset = DatasetTrainTicket2024(
            data_dir=self.data_config.data_dir,
            dates=self.data_config.test_dates,
            node_feature_selector=self.data_config.nfeat_select,
            edge_reverse=False,
            add_self_loop=False,
            max_samples=max_samples,
            failure_types=self.data_config.failure_types,
            failure_duration=self.data_config.failure_duration,
            is_mask=False,
            process_miss=self.data_config.process_miss,
            process_extreme=self.data_config.process_extreme,
            k_sigma=self.data_config.k_sigma,
            use_split_info=False,
        )
        
        logger.info(f'测试数据集加载完成: {len(self.test_dataset)} 个样本')
        return self.test_dataset
    
    def scale_dataset(self, dataset: myDGLDataset, scalers_dir: Optional[str] = None) -> myDGLDataset:
        """
        缩放数据集特征。
        
        Args:
            dataset: 数据集
            scalers_dir: scaler保存目录（如果提供，会加载已有scaler）
        
        Returns:
            缩放后的数据集
        """
        logger.info('缩放数据集特征...')
        
        # 如果提供了scaler目录，尝试加载已有scaler
        if scalers_dir and os.path.exists(scalers_dir):
            from utils import load_node_scalers, load_edge_scalers  # pyright: ignore[reportMissingImports]
            try:
                node_scalers = load_node_scalers(scalers_dir, self.data_config.node_scale_type)
                edge_scalers = load_edge_scalers(scalers_dir, self.data_config.edge_scale_type)
                
                logger.info('加载已有scaler')
                dataset.node_scale(
                    node_scalers=node_scalers,
                    attr='feat',
                    node_scale_type=self.data_config.node_scale_type
                )
                dataset.edge_scale(
                    edge_scalers=edge_scalers,
                    attr='feat',
                    edge_scale_type=self.data_config.edge_scale_type
                )
                return dataset
            except Exception as e:
                logger.warning(f'加载scaler失败: {e}，将重新计算')
        
        # 否则，使用训练数据集进行缩放（如果测试集，需要先有训练集）
        if self.train_dataset is None:
            logger.warning('训练数据集未加载，使用当前数据集本身进行缩放')
            scale_dataset = dataset
        else:
            scale_dataset = self.train_dataset
        
        # 执行缩放
        dataset.node_scale(attr='feat', node_scale_type=self.data_config.node_scale_type)
        dataset.edge_scale(attr='feat', edge_scale_type=self.data_config.edge_scale_type)
        
        logger.info('数据集缩放完成')
        return dataset
    
    def prepare_data_for_rca(
        self,
        dataset: myDGLDataset,
        trace_dir: Optional[str] = None,
        metric_dir: Optional[str] = None,
        use_service_timeseries: bool = False,
        tau_max: int = 10,
    ) -> Dict:
        """
        为RCA准备数据。
        
        Args:
            dataset: 数据集
            trace_dir: Trace数据目录（如果提供，将构建服务级时间序列）
            metric_dir: Metric数据目录（如果提供，将构建服务级时间序列）
            use_service_timeseries: 是否使用新的服务级时间序列构建方法（用于时滞感知RCA）
            tau_max: 最大时滞（用于历史窗口编码）
        
        Returns:
            包含以下键的字典:
            - graphs: List[DGLGraph] 图列表
            - stacked_nfeat: Dict[str, th.Tensor] 堆叠特征
            - labels: List[Dict] 标签列表
            - groundtruths: List[set] 真实根因列表
            - nan_nodes: Dict[str, th.Tensor] NaN节点掩码
            - data_stats: Dict 数据统计（需要单独计算）
            - service_timeseries: (可选) (T, N, F) 服务级时间序列
            - service_list: (可选) 服务列表
            - X_norm: (可选) 正常基线特征
            - G_trace_edges: (可选) Trace拓扑边列表
        """
        logger.info('准备RCA数据...')
        
        # 获取堆叠特征
        stacked_nfeat = dataset.get_stacked_nfeat()
        
        # 获取标签
        labels = dataset.get_labels()
        
        # 获取真实根因
        groundtruths = dataset.get_groundtruths()
        
        # 获取NaN节点掩码
        nan_nodes = dataset.get_nan_nodes()
        
        # 获取图列表
        graphs = [dataset[i][0] for i in range(len(dataset))]
        
        data = {
            'graphs': graphs,
            'stacked_nfeat': stacked_nfeat,
            'labels': labels,
            'groundtruths': groundtruths,
            'nan_nodes': nan_nodes,
        }
        
        # 如果提供了 trace_dir 和 metric_dir，构建服务级时间序列（用于时滞感知RCA）
        if use_service_timeseries and trace_dir and metric_dir:
            logger.info('构建服务级时间序列（用于时滞感知RCA Stage 1）...')
            
            try:
                # 初始化解析器
                trace_parser = TraceParser(
                    trace_dir=trace_dir,
                    min_call_frequency=5,
                    time_window=60
                )
                
                metric_aggregator = MetricAggregator(
                    metric_dir=metric_dir,
                    sampling_interval=10
                )
                
                # 解析 Trace 和 Metric
                trace_df, call_counts = trace_parser.parse_traces()
                metric_df = metric_aggregator.load_prometheus_metrics()
                
                if not trace_df.empty and not metric_df.empty:
                    # 构建服务级时间序列
                    X_tensor, service_list, X_norm = build_service_timeseries(
                        trace_parser=trace_parser,
                        metric_aggregator=metric_aggregator,
                        trace_df=trace_df,
                        call_counts=call_counts,
                        metric_df=metric_df,
                        tau_max=tau_max,
                        time_window=60
                    )
                    
                    # 获取 Trace 拓扑边列表（用于约束 PCMCIplus）
                    _, edge_list = trace_parser.get_trace_topology(call_counts)
                    
                    # 添加到返回数据
                    data['service_timeseries'] = X_tensor  # (T, N, F)
                    data['service_list'] = service_list
                    data['X_norm'] = X_norm  # {service_name: (F,) Tensor}
                    data['G_trace_edges'] = edge_list  # [(u, v), ...]
                    
                    logger.info(f'服务级时间序列构建完成: shape={X_tensor.shape}, '
                              f'服务数={len(service_list)}, 边数={len(edge_list)}')
                else:
                    logger.warning('Trace 或 Metric 数据为空，跳过服务级时间序列构建')
            except Exception as e:
                logger.warning(f'构建服务级时间序列失败: {e}，将使用原有方法')
        
        logger.info(f'RCA数据准备完成: {len(graphs)} 个图, {len(labels)} 个标签')
        return data


class AIOps2025DataPreprocessor:
    """AIOps2025 数据预处理器（简化版）"""

    def __init__(self, config):
        """
        Args:
            config: LADSConfig对象
        """
        self.config = config
        self.data_config = config.data
        self.rca_config = config.rca

        self.train_dataset: Optional[myDGLDataset] = None
        self.test_dataset: Optional[myDGLDataset] = None

    def load_train_dataset(self, max_samples: int = 1e9) -> Optional[myDGLDataset]:
        """AIOps2025 当前不使用训练集（只做测试），直接返回 None。"""
        logger.info('AIOps2025 当前未定义训练集，跳过训练数据加载')
        self.train_dataset = None
        return None

    def load_test_dataset(self, max_samples: int = 1e9) -> myDGLDataset:
        """加载 AIOps2025 测试数据集"""
        logger.info('加载 AIOps2025 测试数据集...')

        # data_dir 在 AIOps 场景下指向 output 根目录（与 main.py 中的 data_dir 对应）
        output_root = self.data_config.data_dir
        self.test_dataset = DatasetAIOps2025(
            output_root=output_root,
            dates=self.data_config.test_dates,
            max_samples=int(max_samples),
            failure_types=self.data_config.failure_types,
            failure_duration=self.data_config.failure_duration,
            is_mask=False,
            process_miss=self.data_config.process_miss,
            process_extreme=self.data_config.process_extreme,
            k_sigma=self.data_config.k_sigma,
            use_split_info=False,
        )

        logger.info(f'AIOps2025 测试数据集加载完成: {len(self.test_dataset)} 个样本')
        return self.test_dataset

    def scale_dataset(self, dataset: myDGLDataset, scalers_dir: Optional[str] = None) -> myDGLDataset:
        """AIOps2025 简化处理：当前不做特征缩放，直接返回"""
        logger.info('AIOps2025 暂不进行特征缩放，直接返回原始数据集')
        return dataset

    def prepare_data_for_rca(self, dataset: myDGLDataset) -> Dict:
        """为 AIOps2025 准备 RCA 所需的数据结构"""
        logger.info('准备 AIOps2025 RCA 数据...')

        # 默认：按“天”构图（一个日期一个图）
        stacked_nfeat = dataset.get_stacked_nfeat()
        labels = dataset.get_labels()
        groundtruths = dataset.get_groundtruths()
        nan_nodes = dataset.get_nan_nodes()
        graphs = [dataset[i][0] for i in range(len(dataset))]

        # AIOps2025：若 Dataset 提供 per_row_node_feats，则展开为“按事件(groundtruth 行)”的样本，
        # 让特征对齐每条故障事件的 timestamp，从而重点提升 top-1。
        if (
            hasattr(dataset, "per_row_labels")
            and hasattr(dataset, "per_row_groundtruths")
            and hasattr(dataset, "per_row_graph_indices")
            and hasattr(dataset, "per_row_node_feats")
            and getattr(dataset, "per_row_labels", None)
            and getattr(dataset, "per_row_node_feats", None)
        ):
            try:
                from copy import deepcopy

                per_row_labels = list(getattr(dataset, "per_row_labels"))
                per_row_groundtruths = list(getattr(dataset, "per_row_groundtruths"))
                per_row_graph_indices = list(getattr(dataset, "per_row_graph_indices"))
                per_row_node_feats = list(getattr(dataset, "per_row_node_feats"))

                graphs_row = []
                feats_row = []

                for i, gidx in enumerate(per_row_graph_indices):
                    if not (0 <= gidx < len(dataset.graphs)):
                        continue
                    g_base = dataset.graphs[gidx]
                    g_evt = deepcopy(g_base)

                    # 事件级特征：(N, 8)，拼接入/出度 -> (N, 10)
                    in_deg = g_evt.in_degrees(etype=("pod", "calls", "pod")).float()
                    out_deg = g_evt.out_degrees(etype=("pod", "calls", "pod")).float()
                    mfeat = per_row_node_feats[i].float()
                    if mfeat.dim() != 2 or mfeat.shape[0] != g_evt.num_nodes("pod") or mfeat.shape[1] != 8:
                        # 尺寸不匹配则回退用原始特征
                        feat = g_evt.ndata["feat"]
                    else:
                        feat = th.cat(
                            [in_deg.unsqueeze(-1), out_deg.unsqueeze(-1), mfeat],
                            dim=-1,
                        )
                        g_evt.ndata["feat"] = feat

                    graphs_row.append(g_evt)
                    feats_row.append(feat)

                if graphs_row and feats_row:
                    graphs = graphs_row
                    stacked_nfeat = {"pod": th.stack(feats_row, dim=0)}
                    labels = per_row_labels[: len(graphs)]
                    groundtruths = per_row_groundtruths[: len(graphs)]
                    # nan_nodes：按新特征重新生成（简单：全 False）
                    nan_nodes = {"pod": th.zeros((len(graphs), stacked_nfeat["pod"].shape[1]), dtype=th.bool)}
                    logger.info(f"AIOps2025 启用按事件展开：{len(graphs)} 条事件样本")
            except Exception as e:
                logger.warning(f"AIOps2025 按事件展开失败，回退按天：{e}")

        data = {
            'graphs': graphs,
            'stacked_nfeat': stacked_nfeat,
            'labels': labels,
            'groundtruths': groundtruths,
            'nan_nodes': nan_nodes,
        }

        # 构建服务级时间序列（用于 Stage 1 PCMCIplus）
        # 从 dataset 中获取日期信息，构建 trace 和 metric 路径
        try:
            if hasattr(dataset, 'dates') and dataset.dates:
                date = dataset.dates[0]  # 使用第一个日期
                date_suffix = date.replace('-', '')

                # 构建路径
                output_root = Path(self.data_config.data_dir)
                trace_dir = output_root / f"aiops2025_{date_suffix}" / "trace_propagation_aug_flatcsv"
                metric_dir = output_root / f"aiops2025_{date_suffix}" / "metric_propagation_aug"

                if trace_dir.exists() and metric_dir.exists():
                    from data_pipeline import TraceParser, MetricAggregator, build_service_timeseries

                    # 初始化解析器
                    trace_parser = TraceParser(
                        trace_dir=str(trace_dir),
                        min_call_frequency=5,
                        time_window=60
                    )

                    metric_aggregator = MetricAggregator(
                        metric_dir=str(metric_dir),
                        sampling_interval=10
                    )

                    # 解析 Trace 和 Metric
                    trace_df, call_counts = trace_parser.parse_traces()
                    metric_df = metric_aggregator.load_prometheus_metrics()

                    if not trace_df.empty and not metric_df.empty:
                        # 构建服务级时间序列
                        X_tensor, service_list, X_norm = build_service_timeseries(
                            trace_parser=trace_parser,
                            metric_aggregator=metric_aggregator,
                            trace_df=trace_df,
                            call_counts=call_counts,
                            metric_df=metric_df,
                            tau_max=self.rca_config.tau_max,
                            time_window=60
                        )

                        # 获取 Trace 拓扑边列表（用于约束 PCMCIplus）
                        _, edge_list = trace_parser.get_trace_topology(call_counts)

                        # 添加到返回数据
                        data['service_timeseries'] = X_tensor  # (T, N, F)
                        data['service_list'] = service_list
                        data['X_norm'] = X_norm  # {service_name: (F,) Tensor}
                        data['G_trace_edges'] = edge_list  # [(u, v), ...]

                        logger.info(f'服务级时间序列构建完成: shape={X_tensor.shape}, '
                                  f'服务数={len(service_list)}, 边数={len(edge_list)}')
                    else:
                        logger.warning('Trace 或 Metric 数据为空，跳过服务级时间序列构建')
                else:
                    logger.warning(f'Trace 或 Metric 目录不存在: {trace_dir} 或 {metric_dir}')
            else:
                logger.warning('Dataset 没有 dates 属性，跳过服务级时间序列构建')
        except Exception as e:
            logger.warning(f'构建服务级时间序列失败: {e}，将使用原有方法')
            import traceback
            logger.debug(traceback.format_exc())

        logger.info(f'AIOps2025 RCA 数据准备完成: {len(graphs)} 个图, {len(labels)} 个标签')
        return data


class RCAbenchDataPreprocessor:
    """RCAbench 数据预处理器（scenario 级别，依赖 preprocess_rcabench.py 生成的目录）"""

    def __init__(self, config):
        self.config = config
        self.data_config = config.data
        self.rca_config = config.rca
        self.train_dataset: Optional[myDGLDataset] = None
        self.test_dataset: Optional[myDGLDataset] = None

    def load_train_dataset(self, max_samples: int = 1e9) -> Optional[myDGLDataset]:
        logger.info("RCAbench 当前不使用训练集，跳过训练数据加载")
        self.train_dataset = None
        return None

    def load_test_dataset(self, max_samples: int = 1e9) -> myDGLDataset:
        logger.info("加载 RCAbench 测试数据集...")
        # data_dir 指向 preprocess 输出根目录（包含 scenarios/）
        preprocessed_root = self.data_config.data_dir
        # 复用 test_dates 作为“scenario 列表”（若为空则加载全部）
        scenarios = list(self.data_config.test_dates) if self.data_config.test_dates else None
        self.test_dataset = DatasetRCAbench(
            preprocessed_root=preprocessed_root,
            scenarios=scenarios,
            max_samples=int(max_samples),
            failure_types=self.data_config.failure_types,
            add_self_loop=True,
        )
        logger.info(f"RCAbench 测试数据集加载完成: {len(self.test_dataset)} 个样本")
        return self.test_dataset

    def scale_dataset(self, dataset: myDGLDataset, scalers_dir: Optional[str] = None) -> myDGLDataset:
        logger.info("RCAbench 暂不进行特征缩放，直接返回原始数据集")
        return dataset

    @staticmethod
    def _normalize_failure_type_name(label: Optional[Dict]) -> str:
        if not isinstance(label, dict):
            return "unknown"
        return str(label.get("failure_type", "") or "").strip().lower() or "unknown"

    @staticmethod
    def _safe_column_minmax(feat: th.Tensor) -> th.Tensor:
        if feat.numel() == 0:
            return feat
        col_min = feat.min(dim=0, keepdim=True).values
        col_max = feat.max(dim=0, keepdim=True).values
        denom = col_max - col_min
        denom = th.where(denom < 1e-12, th.ones_like(denom), denom)
        return (feat - col_min) / denom

    def _augment_rcabench_stage1_features(
        self,
        stacked_nfeat: Dict[str, th.Tensor],
        labels: List[Dict],
    ) -> Dict[str, th.Tensor]:
        """Append RCAbench fault-family-aware Stage-1 features on top of the base 20 dims."""
        pod_feat_seq = stacked_nfeat.get("pod")
        if pod_feat_seq is None or pod_feat_seq.dim() != 3 or pod_feat_seq.shape[-1] < 20:
            return stacked_nfeat
        if int(pod_feat_seq.shape[-1]) >= 36:
            logger.info(
                f"RCAbench Stage 1 已包含原始增强特征，跳过派生增强: pod 维度 {pod_feat_seq.shape[-1]}"
            )
            return stacked_nfeat

        augmented_slices: List[th.Tensor] = []
        for sample_idx in range(pod_feat_seq.shape[0]):
            feat = pod_feat_seq[sample_idx].float()
            label = labels[sample_idx] if sample_idx < len(labels) else {}
            failure_type = self._normalize_failure_type_name(label)

            feat_last = feat[:, 2:10]
            feat_delta = feat[:, 10:18]
            feat_trace = feat[:, 18:20]

            cpu_last, mem_last, disk_last, net_in_last, net_out_last, workload_last, latency_last, error_last = [feat_last[:, i] for i in range(8)]
            cpu_delta, mem_delta, disk_delta, net_in_delta, net_out_delta, workload_delta, latency_delta, error_delta = [feat_delta[:, i] for i in range(8)]
            trace_latency = feat_trace[:, 0]
            trace_drop = feat_trace[:, 1]

            stress_gate = 1.0 if any(k in failure_type for k in ("stress", "cpu", "memory")) else 0.0
            network_gate = 1.0 if any(k in failure_type for k in ("partition", "loss", "bandwidth", "network")) else 0.0
            pod_gate = 1.0 if any(k in failure_type for k in ("pod-failure", "container-kill")) else 0.0
            mysql_gate = 1.0 if "mysql" in failure_type else 0.0

            stress_pressure = cpu_last + mem_last + workload_last + cpu_delta + mem_delta + workload_delta
            stress_saturation = th.stack([cpu_last, mem_last, workload_last, latency_last], dim=1).max(dim=1).values
            stress_backlog = workload_last + latency_last + workload_delta

            network_volume = net_in_last + net_out_last
            network_shift = net_in_delta + net_out_delta + latency_delta
            network_instability = network_shift + error_delta + trace_latency + trace_drop

            pod_failure_signal = error_last + error_delta + trace_drop
            pod_failure_burst = cpu_delta + mem_delta + disk_delta
            pod_failure_latency = latency_delta + trace_latency

            mysql_pressure = latency_last + workload_last + cpu_last + mem_last
            mysql_instability = latency_delta + error_delta + trace_latency
            mysql_backlog = workload_delta + latency_last + trace_drop

            extra_feat = th.stack(
                [
                    th.full_like(stress_pressure, stress_gate),
                    stress_pressure * stress_gate,
                    stress_saturation * stress_gate,
                    stress_backlog * stress_gate,
                    th.full_like(network_volume, network_gate),
                    network_volume * network_gate,
                    network_shift * network_gate,
                    network_instability * network_gate,
                    th.full_like(pod_failure_signal, pod_gate),
                    pod_failure_signal * pod_gate,
                    pod_failure_burst * pod_gate,
                    pod_failure_latency * pod_gate,
                    th.full_like(mysql_pressure, mysql_gate),
                    mysql_pressure * mysql_gate,
                    mysql_instability * mysql_gate,
                    mysql_backlog * mysql_gate,
                ],
                dim=1,
            )
            extra_feat = self._safe_column_minmax(extra_feat)
            augmented_slices.append(th.cat([feat, extra_feat], dim=1))

        stacked_nfeat = dict(stacked_nfeat)
        stacked_nfeat["pod"] = th.stack(augmented_slices, dim=0)
        logger.info(
            f"RCAbench Stage 1 特征增强完成: pod 维度 {pod_feat_seq.shape[-1]} -> {stacked_nfeat['pod'].shape[-1]}"
        )
        return stacked_nfeat

    def prepare_data_for_rca(self, dataset: myDGLDataset) -> Dict:
        logger.info("准备 RCAbench RCA 数据...")
        stacked_nfeat = dataset.get_stacked_nfeat()
        labels = dataset.get_labels()
        groundtruths = dataset.get_groundtruths()
        nan_nodes = dataset.get_nan_nodes()
        graphs = [dataset[i][0] for i in range(len(dataset))]
        data = {
            "graphs": graphs,
            "stacked_nfeat": stacked_nfeat,
            "labels": labels,
            "groundtruths": groundtruths,
            "nan_nodes": nan_nodes,
        }
        if hasattr(dataset, "get_stage1_channel_aux"):
            try:
                data["stage1_channel_aux"] = dataset.get_stage1_channel_aux()
            except Exception:
                logger.warning("RCAbench Stage 1 辅助特征读取失败，回退为默认通道特征")
        logger.info(f"RCAbench RCA 数据准备完成: {len(graphs)} 个图, {len(labels)} 个标签")
        return data
    
    def compute_data_statistics(
        self,
        dataset: myDGLDataset,
        model,
        device: str = 'cpu',
    ) -> Dict:
        """
        计算数据统计信息（用于异常检测和RCA）。
        
        Args:
            dataset: 数据集
            model: 模型（用于预测）
            device: 设备
        
        Returns:
            数据统计字典，包含 mean 和 cov_inv
        """
        logger.info('计算数据统计信息...')
        
        # 这里简化实现，实际应该使用模型预测
        # 参考 run.py 中的 compute_statistics_for_rca
        from utils import load_stats  # pyright: ignore[reportMissingImports]
        
        # 尝试从文件加载（如果已存在）
        stats_file = os.path.join(self.config.output_dir, 'data_stats_pred_and_mean.json')
        if os.path.exists(stats_file):
            logger.info('加载已有数据统计文件')
            return load_stats(stats_file, device)
        
        # 否则需要计算（这里简化，实际需要模型推理）
        logger.warning('数据统计文件不存在，使用简化计算')
        
        stacked_nfeat = dataset.get_stacked_nfeat()
        data_stats = {'mean': {}, 'cov_inv': {}}
        
        for ntype in ['api', 'pod']:
            if ntype in stacked_nfeat:
                # 计算均值
                feats = stacked_nfeat[ntype]  # (T, N, F)
                mean_feat = th.mean(feats, dim=0)  # (N, F)
                data_stats['mean'][ntype] = mean_feat.to(device)
                
                # 简化：使用单位矩阵作为协方差逆矩阵
                num_nodes = feats.shape[1]
                num_feats = feats.shape[2]
                cov_inv = th.eye(num_feats).unsqueeze(0).repeat(num_nodes, 1, 1).to(device)
                data_stats['cov_inv'][ntype] = cov_inv
        
        logger.info('数据统计信息计算完成')
        return data_stats
    
    def build_pod_features(
        self,
        dataset: myDGLDataset,
        stacked_nfeat: Dict[str, th.Tensor],
        tau_max: int,
    ) -> Tuple[Dict, Dict, Dict]:
        """
        构建Pod特征数据（用于Stage 2）。
        
        Args:
            dataset: 数据集
            stacked_nfeat: 堆叠特征
            tau_max: 最大时滞
        
        Returns:
            (pod_feats, norm_pod_feats, downstream_feats)
            - pod_feats: {service_id: {pod_id: {lag: features}}}
            - norm_pod_feats: {service_id: normalized_features}
            - downstream_feats: {service_id: downstream_features}
        """
        logger.info('构建Pod特征数据...')
        
        # 简化实现：从dataset中提取Pod特征
        pod_feats = {}  # {service_id: {pod_id: {lag: features}}}
        norm_pod_feats = {}  # {service_id: normalized_features}
        downstream_feats = {}  # {service_id: downstream_features}
        
        if 'pod' not in stacked_nfeat:
            logger.warning('未找到pod特征，返回空字典')
            return pod_feats, norm_pod_feats, downstream_feats
        
        pod_feat_seq = stacked_nfeat['pod']  # (T, N_pod, F)
        
        # 获取Pod到服务的映射（如果有）
        # 这里简化：假设pod节点ID就是服务ID
        # 实际应该从dataset中获取pod_to_service映射
        
        # 对于每个pod，提取历史特征
        num_pods = pod_feat_seq.shape[1]
        
        for pod_id in range(num_pods):
            # 假设pod_id对应service_id（简化）
            service_id = pod_id
            
            if service_id not in pod_feats:
                pod_feats[service_id] = {}
                # 使用nanmean避免NaN值影响
                pod_feat_mean = th.nanmean(pod_feat_seq[:, pod_id, :], dim=0)
                # 如果仍有NaN，使用0填充
                pod_feat_mean = th.where(th.isnan(pod_feat_mean), th.zeros_like(pod_feat_mean), pod_feat_mean)
                norm_pod_feats[service_id] = pod_feat_mean
            
            # 提取历史特征
            hist_feats = {}
            T = pod_feat_seq.shape[0]
            for lag in range(tau_max + 1):
                if lag < T:
                    hist_feats[lag] = pod_feat_seq[T - 1 - lag, pod_id, :]
                else:
                    hist_feats[lag] = pod_feat_seq[0, pod_id, :]  # 使用第一个时刻
            
            pod_feats[service_id][pod_id] = hist_feats
        
        # downstream_feats（简化：使用api特征）
        if 'api' in stacked_nfeat:
            api_feat_seq = stacked_nfeat['api']  # (T, N_api, F)
            num_apis = api_feat_seq.shape[1]
            for service_id in range(min(num_apis, num_pods)):
                downstream_feats[service_id] = api_feat_seq[-1, service_id, :]  # 使用最后一个时刻
        
        logger.info(f'Pod特征构建完成: {len(pod_feats)} 个服务, {sum(len(pods) for pods in pod_feats.values())} 个Pod')
        return pod_feats, norm_pod_feats, downstream_feats
