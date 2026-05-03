"""Dataset preprocessors used by the public RippleRCA release."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch as th  # pyright: ignore[reportMissingImports]
from data_pipeline import (
    MetricAggregator,
    TraceParser,
    build_service_timeseries,
)  # pyright: ignore[reportMissingImports]
from dataset import myDGLDataset  # pyright: ignore[reportMissingImports]
from dataset_aiops_2025 import DatasetAIOps2025  # pyright: ignore[reportMissingImports]
from dataset_rcabench import DatasetRCAbench  # pyright: ignore[reportMissingImports]
from log import Logger  # pyright: ignore[reportMissingImports]

logger = Logger(__name__)


def _stack_graphs(dataset: myDGLDataset) -> List[Any]:
    return [dataset[i][0] for i in range(len(dataset))]


class AIOps2025DataPreprocessor:
    def __init__(self, config) -> None:
        self.config = config
        self.data_config = config.data
        self.rca_config = config.rca
        self.test_dataset: Optional[myDGLDataset] = None

    def load_train_dataset(self, max_samples: int = 10**9) -> Optional[myDGLDataset]:
        return None

    def load_test_dataset(self, max_samples: int = 10**9) -> myDGLDataset:
        self.test_dataset = DatasetAIOps2025(
            output_root=self.data_config.data_dir,
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
        return self.test_dataset

    def scale_dataset(
        self, dataset: myDGLDataset, scalers_dir: Optional[str] = None
    ) -> myDGLDataset:
        return dataset

    def _build_event_level_graphs(self, dataset: myDGLDataset) -> Tuple[
        List[Any],
        Dict[str, th.Tensor],
        List[Dict],
        List[set],
        Dict[str, th.Tensor],
        List[int],
    ]:
        per_row_labels = list(getattr(dataset, "per_row_labels"))
        per_row_groundtruths = list(getattr(dataset, "per_row_groundtruths"))
        per_row_graph_indices = list(getattr(dataset, "per_row_graph_indices"))
        per_row_node_feats = list(getattr(dataset, "per_row_node_feats"))
        graphs: List[Any] = []
        feats_row: List[th.Tensor] = []
        labels: List[Dict] = []
        groundtruths: List[set] = []
        for row_index, graph_index in enumerate(per_row_graph_indices):
            if not (0 <= graph_index < len(dataset.graphs)):
                continue
            graph = deepcopy(dataset.graphs[graph_index])
            metric_feat = per_row_node_feats[row_index].float()
            if (
                metric_feat.dim() == 2
                and metric_feat.shape[0] == graph.num_nodes("pod")
                and metric_feat.shape[1] == 8
            ):
                in_deg = (
                    graph.in_degrees(etype=("pod", "calls", "pod"))
                    .float()
                    .unsqueeze(-1)
                )
                out_deg = (
                    graph.out_degrees(etype=("pod", "calls", "pod"))
                    .float()
                    .unsqueeze(-1)
                )
                feat = th.cat([in_deg, out_deg, metric_feat], dim=-1)
                graph.ndata["feat"] = feat
            else:
                feat = graph.ndata["feat"]
            graphs.append(graph)
            feats_row.append(feat)
            labels.append(per_row_labels[row_index])
            groundtruths.append(per_row_groundtruths[row_index])
        stacked_nfeat = (
            {"pod": th.stack(feats_row, dim=0)}
            if feats_row
            else dataset.get_stacked_nfeat()
        )
        nan_nodes = {
            "pod": th.zeros(
                (len(graphs), stacked_nfeat["pod"].shape[1]),
                dtype=th.bool,
            )
        }
        return (
            graphs,
            stacked_nfeat,
            labels,
            groundtruths,
            nan_nodes,
            per_row_graph_indices[: len(graphs)],
        )

    def _load_service_level_inputs(self, dataset: myDGLDataset) -> Dict[str, Any]:
        if not getattr(dataset, "dates", None):
            return {}
        date = dataset.dates[0]
        date_suffix = str(date).replace("-", "")
        output_root = Path(self.data_config.data_dir)
        trace_dir = (
            output_root / f"aiops2025_{date_suffix}" / "trace_propagation_aug_flatcsv"
        )
        metric_dir = output_root / f"aiops2025_{date_suffix}" / "metric_propagation_aug"
        if not trace_dir.exists() or not metric_dir.exists():
            return {}
        trace_parser = TraceParser(
            trace_dir=str(trace_dir),
            min_call_frequency=5,
            time_window=60,
        )
        metric_aggregator = MetricAggregator(
            metric_dir=str(metric_dir),
            sampling_interval=10,
        )
        trace_df, call_counts = trace_parser.parse_traces()
        metric_df = metric_aggregator.load_prometheus_metrics()
        if trace_df.empty or metric_df.empty:
            return {}
        timeseries, service_list, x_norm = build_service_timeseries(
            trace_parser=trace_parser,
            metric_aggregator=metric_aggregator,
            trace_df=trace_df,
            call_counts=call_counts,
            metric_df=metric_df,
            tau_max=self.rca_config.tau_max,
            time_window=60,
        )
        _, edge_list = trace_parser.get_trace_topology(call_counts)
        return {
            "service_timeseries": timeseries,
            "service_list": service_list,
            "X_norm": x_norm,
            "G_trace_edges": edge_list,
        }

    def prepare_data_for_rca(self, dataset: myDGLDataset) -> Dict[str, Any]:
        graphs = _stack_graphs(dataset)
        stacked_nfeat = dataset.get_stacked_nfeat()
        labels = dataset.get_labels()
        groundtruths = dataset.get_groundtruths()
        nan_nodes = dataset.get_nan_nodes()
        per_row_graph_indices = None
        if (
            hasattr(dataset, "per_row_labels")
            and hasattr(dataset, "per_row_groundtruths")
            and hasattr(dataset, "per_row_graph_indices")
            and hasattr(dataset, "per_row_node_feats")
            and getattr(dataset, "per_row_labels", None)
            and getattr(dataset, "per_row_node_feats", None)
        ):
            (
                graphs,
                stacked_nfeat,
                labels,
                groundtruths,
                nan_nodes,
                per_row_graph_indices,
            ) = self._build_event_level_graphs(dataset)
        data: Dict[str, Any] = {
            "graphs": graphs,
            "stacked_nfeat": stacked_nfeat,
            "labels": labels,
            "groundtruths": groundtruths,
            "nan_nodes": nan_nodes,
        }
        if per_row_graph_indices is not None:
            data["per_row_graph_indices"] = per_row_graph_indices
        data.update(self._load_service_level_inputs(dataset))
        return data

    def build_pod_features(
        self,
        dataset: myDGLDataset,
        stacked_nfeat: Dict[str, th.Tensor],
        tau_max: int,
    ) -> Tuple[Dict[Any, Any], Dict[Any, th.Tensor], Dict[Any, th.Tensor]]:
        pod_feats: Dict[Any, Any] = {}
        norm_pod_feats: Dict[Any, th.Tensor] = {}
        downstream_feats: Dict[Any, th.Tensor] = {}
        pod_feat_seq = stacked_nfeat.get("pod")
        if pod_feat_seq is None or pod_feat_seq.dim() != 3:
            return pod_feats, norm_pod_feats, downstream_feats
        num_pods = int(pod_feat_seq.shape[1])
        seq_len = int(pod_feat_seq.shape[0])
        for pod_id in range(num_pods):
            service_id = pod_id
            pod_mean = th.nanmean(pod_feat_seq[:, pod_id, :], dim=0)
            pod_mean = th.where(th.isnan(pod_mean), th.zeros_like(pod_mean), pod_mean)
            norm_pod_feats[service_id] = pod_mean
            hist_feats: Dict[int, th.Tensor] = {}
            for lag in range(tau_max + 1):
                src_idx = max(seq_len - 1 - lag, 0)
                feat_lag = pod_feat_seq[src_idx, pod_id, :]
                feat_lag = th.where(
                    th.isnan(feat_lag), th.zeros_like(feat_lag), feat_lag
                )
                hist_feats[lag] = feat_lag
            pod_feats.setdefault(service_id, {})[pod_id] = hist_feats
            downstream_feats[service_id] = hist_feats[0]
        return pod_feats, norm_pod_feats, downstream_feats


class RCAbenchDataPreprocessor:
    def __init__(self, config) -> None:
        self.config = config
        self.data_config = config.data
        self.rca_config = config.rca
        self.test_dataset: Optional[myDGLDataset] = None

    def load_train_dataset(self, max_samples: int = 10**9) -> Optional[myDGLDataset]:
        return None

    def load_test_dataset(self, max_samples: int = 10**9) -> myDGLDataset:
        scenarios = (
            list(self.data_config.test_dates) if self.data_config.test_dates else None
        )
        self.test_dataset = DatasetRCAbench(
            preprocessed_root=self.data_config.data_dir,
            scenarios=scenarios,
            max_samples=int(max_samples),
            failure_types=self.data_config.failure_types,
            add_self_loop=True,
        )
        return self.test_dataset

    def scale_dataset(
        self, dataset: myDGLDataset, scalers_dir: Optional[str] = None
    ) -> myDGLDataset:
        return dataset

    @staticmethod
    def _normalize_failure_type_name(label: Optional[Dict[str, Any]]) -> str:
        if not isinstance(label, dict):
            return "unknown"
        return str(label.get("failure_type", "") or "").strip().lower() or "unknown"

    @staticmethod
    def _safe_column_minmax(feat: th.Tensor) -> th.Tensor:
        if feat.numel() == 0:
            return feat
        col_min = feat.min(dim=0, keepdim=True).values
        col_max = feat.max(dim=0, keepdim=True).values
        denom = th.where(
            (col_max - col_min) < 1e-12, th.ones_like(col_max), col_max - col_min
        )
        return (feat - col_min) / denom

    def _augment_stage1_features(
        self,
        stacked_nfeat: Dict[str, th.Tensor],
        labels: List[Dict[str, Any]],
    ) -> Dict[str, th.Tensor]:
        pod_feat_seq = stacked_nfeat.get("pod")
        if (
            pod_feat_seq is None
            or pod_feat_seq.dim() != 3
            or pod_feat_seq.shape[-1] < 20
        ):
            return stacked_nfeat
        if int(pod_feat_seq.shape[-1]) >= 36:
            return stacked_nfeat
        augmented_slices: List[th.Tensor] = []
        for sample_idx in range(pod_feat_seq.shape[0]):
            feat = pod_feat_seq[sample_idx].float()
            failure_type = self._normalize_failure_type_name(
                labels[sample_idx] if sample_idx < len(labels) else {}
            )
            feat_last = feat[:, 2:10]
            feat_delta = feat[:, 10:18]
            feat_trace = feat[:, 18:20]
            (
                cpu_last,
                mem_last,
                disk_last,
                net_in_last,
                net_out_last,
                workload_last,
                latency_last,
                error_last,
            ) = [feat_last[:, i] for i in range(8)]
            (
                cpu_delta,
                mem_delta,
                disk_delta,
                net_in_delta,
                net_out_delta,
                workload_delta,
                latency_delta,
                error_delta,
            ) = [feat_delta[:, i] for i in range(8)]
            trace_latency = feat_trace[:, 0]
            trace_drop = feat_trace[:, 1]
            stress_gate = (
                1.0
                if any(key in failure_type for key in ("stress", "cpu", "memory"))
                else 0.0
            )
            network_gate = (
                1.0
                if any(
                    key in failure_type
                    for key in ("partition", "loss", "bandwidth", "network")
                )
                else 0.0
            )
            pod_gate = (
                1.0
                if any(key in failure_type for key in ("pod-failure", "container-kill"))
                else 0.0
            )
            mysql_gate = 1.0 if "mysql" in failure_type else 0.0
            stress_pressure = (
                cpu_last
                + mem_last
                + workload_last
                + cpu_delta
                + mem_delta
                + workload_delta
            )
            stress_saturation = (
                th.stack([cpu_last, mem_last, workload_last, latency_last], dim=1)
                .max(dim=1)
                .values
            )
            stress_backlog = workload_last + latency_last + workload_delta
            network_volume = net_in_last + net_out_last
            network_shift = net_in_delta + net_out_delta + latency_delta
            network_instability = (
                network_shift + error_delta + trace_latency + trace_drop
            )
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
            augmented_slices.append(
                th.cat([feat, self._safe_column_minmax(extra_feat)], dim=1)
            )
        stacked_nfeat = dict(stacked_nfeat)
        stacked_nfeat["pod"] = th.stack(augmented_slices, dim=0)
        return stacked_nfeat

    def prepare_data_for_rca(self, dataset: myDGLDataset) -> Dict[str, Any]:
        labels = dataset.get_labels()
        stacked_nfeat = self._augment_stage1_features(
            dataset.get_stacked_nfeat(), labels
        )
        data: Dict[str, Any] = {
            "graphs": _stack_graphs(dataset),
            "stacked_nfeat": stacked_nfeat,
            "labels": labels,
            "groundtruths": dataset.get_groundtruths(),
            "nan_nodes": dataset.get_nan_nodes(),
        }
        if hasattr(dataset, "get_stage1_channel_aux"):
            try:
                data["stage1_channel_aux"] = dataset.get_stage1_channel_aux()
            except Exception:
                logger.warning(
                    "Failed to load optional Stage 1 auxiliary signals for RCAbench."
                )
        return data

    def build_pod_features(
        self,
        dataset: myDGLDataset,
        stacked_nfeat: Dict[str, th.Tensor],
        tau_max: int,
    ) -> Tuple[Dict[Any, Any], Dict[Any, th.Tensor], Dict[Any, th.Tensor]]:
        pod_feats: Dict[Any, Any] = {}
        norm_pod_feats: Dict[Any, th.Tensor] = {}
        downstream_feats: Dict[Any, th.Tensor] = {}
        pod_feat_seq = stacked_nfeat.get("pod")
        if pod_feat_seq is None or pod_feat_seq.dim() != 3:
            return pod_feats, norm_pod_feats, downstream_feats
        num_pods = int(pod_feat_seq.shape[1])
        seq_len = int(pod_feat_seq.shape[0])
        for pod_id in range(num_pods):
            service_id = pod_id
            pod_mean = th.nanmean(pod_feat_seq[:, pod_id, :], dim=0)
            pod_mean = th.where(th.isnan(pod_mean), th.zeros_like(pod_mean), pod_mean)
            norm_pod_feats[service_id] = pod_mean
            hist_feats: Dict[int, th.Tensor] = {}
            for lag in range(tau_max + 1):
                src_idx = max(seq_len - 1 - lag, 0)
                feat_lag = pod_feat_seq[src_idx, pod_id, :]
                feat_lag = th.where(
                    th.isnan(feat_lag), th.zeros_like(feat_lag), feat_lag
                )
                hist_feats[lag] = feat_lag
            pod_feats.setdefault(service_id, {})[pod_id] = hist_feats
        api_feat_seq = stacked_nfeat.get("api")
        if api_feat_seq is not None and api_feat_seq.dim() == 3:
            num_apis = int(api_feat_seq.shape[1])
            for service_id in range(min(num_apis, num_pods)):
                downstream_feats[service_id] = api_feat_seq[-1, service_id, :]
        return pod_feats, norm_pod_feats, downstream_feats
