from __future__ import annotations
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import torch as th
import dgl
from dataset import myDGLDataset
from log import Logger

logger = Logger(__name__)


class DatasetAIOps2025(myDGLDataset):
    def __init__(
        self,
        output_root: str,
        dates: List[str],
        max_samples: int,
        failure_types: List[str],
        failure_duration: int,
        is_mask: bool,
        process_miss: str,
        process_extreme: bool,
        k_sigma: int,
        use_split_info: bool,
        **kwargs,
    ) -> None:
        self.output_root = Path(output_root)
        self.dates = dates
        self.failure_duration = failure_duration
        super().__init__(
            dataset_name="aiops2025",
            data_dir=str(self.output_root),
            dates=dates,
            node_feature_selector={"pod": ["in_degree", "out_degree", "metric_mean"]},
            edge_reverse=False,
            add_self_loop=True,
            max_samples=max_samples,
            failure_types=failure_types,
            special_failure_types=[],
            failure_duration=failure_duration,
            is_mask=is_mask,
            process_miss=process_miss,
            process_extreme=process_extreme,
            k_sigma=k_sigma,
            use_split_info=use_split_info,
        )

    def pod_to_service(self, pod_name: str) -> str:
        parts = str(pod_name).split("-")
        if len(parts) > 2 and parts[-1].isdigit() and parts[-2].isdigit():
            return "-".join(parts[:-2])
        if len(parts) > 1 and parts[-1].isdigit():
            return "-".join(parts[:-1])
        return str(pod_name)

    def process(self) -> None:
        self.graphs = []
        self.labels = []
        self.groundtruths = []
        self.per_row_labels: List[Dict] = []
        self.per_row_groundtruths: List[set] = []
        self.per_row_node_feats: List[th.Tensor] = []
        self.per_row_graph_indices: List[int] = []
        self.ntypes = ["pod"]
        self.etypes = [("pod", "calls", "pod")]
        _exclude_failure_type_keywords = {"memory"}
        for date in self.dates:
            logger.info(f"Processing AIOps2025 Dates: {date}")
            if "-" in date:
                date_iso = date
            else:
                if len(date) == 8:
                    date_iso = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
                else:
                    date_iso = date
            date_suffix = date.replace("-", "")
            out_dir = self.output_root / f"aiops2025_{date_suffix}"
            trace_dir = out_dir / "trace_propagation_aug_flatcsv"
            metric_dir = out_dir / "metric_propagation_aug"
            gt_csv = (
                out_dir / "groundtruth_csv" / f"groundtruth-aiops2025-{date_iso}.csv"
            )
            if not trace_dir.exists():
                logger.warning(f"The Trace directory does not exist.: {trace_dir}")
                continue
            if not metric_dir.exists():
                logger.warning(f"The Metric directory does not exist.: {metric_dir}")
                continue
            csv_files = sorted(f for f in os.listdir(trace_dir) if f.endswith(".csv"))
            dfs = [pd.read_csv(trace_dir / f) for f in csv_files]
            trace_df_raw = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            if trace_df_raw.empty:
                logger.warning(f"{date} The trace data is empty.")
                continue
            max_rows = 500_000
            if len(trace_df_raw) > max_rows:
                logger.info(
                    f"Trace line number {len(trace_df_raw)}，Sampling {max_rows}"
                )
                trace_df_raw = trace_df_raw.sample(n=max_rows, random_state=42)
            span_to_pod = dict(zip(trace_df_raw["SpanID"], trace_df_raw["PodName"]))
            records = []
            for _, row in trace_df_raw.iterrows():
                ts = pd.to_datetime(int(row["StartTimeUnixNano"]), unit="ns")
                child_pod = row.get("PodName", "")
                child_service = self.pod_to_service(child_pod) if child_pod else ""
                parent_service = ""
                parent_id = row.get("ParentID", "")
                if isinstance(parent_id, str) and parent_id and parent_id != "root":
                    parent_pod = span_to_pod.get(parent_id, "")
                    parent_service = (
                        self.pod_to_service(parent_pod) if parent_pod else ""
                    )
                records.append(
                    {
                        "timestamp": ts.to_pydatetime(),
                        "parent_service": parent_service,
                        "child_service": child_service,
                    }
                )
            from collections import defaultdict

            trace_df = pd.DataFrame(records)
            trace_df["time_window"] = (
                trace_df["timestamp"].astype("datetime64[s]").dt.floor("60s")
            )
            call_counts = defaultdict(int)
            for _, row in trace_df.iterrows():
                if row["parent_service"] and row["child_service"]:
                    key = (row["parent_service"], row["child_service"])
                    call_counts[key] += 1
            call_counts = {k: v for k, v in call_counts.items() if v >= 5}
            services = set()
            for p, c in call_counts.keys():
                services.add(p)
                services.add(c)
            service_list = sorted(services)
            if not service_list:
                logger.warning(f"{date} No valid service call relationship, skip.")
                continue
            service_to_id = {svc: i for i, svc in enumerate(service_list)}
            metric_files = sorted(
                f for f in os.listdir(metric_dir) if f.endswith(".csv")
            )
            all_metrics = [pd.read_csv(metric_dir / mf) for mf in metric_files]
            metric_df = (
                pd.concat(all_metrics, ignore_index=True)
                if all_metrics
                else pd.DataFrame()
            )
            if not metric_df.empty and "object_id" in metric_df.columns:
                metric_df["PodName"] = metric_df["object_id"].astype(str)
                metric_df["pod"] = metric_df["object_id"].astype(str)
                metric_df["service"] = (
                    metric_df["object_id"].astype(str).map(self.pod_to_service)
                )
                metric_df["service_name"] = metric_df["service"]
            metric_time_col = None
            for col in [
                "timestamp",
                "time",
                "TimeStamp",
                "datetime",
                "t",
                "ts",
                "date",
            ]:
                if col in metric_df.columns:
                    metric_time_col = col
                    break
            if metric_time_col is not None and not metric_df.empty:
                ts_raw = metric_df[metric_time_col]
                if np.issubdtype(ts_raw.dtype, np.number):
                    v = pd.to_numeric(ts_raw, errors="coerce")
                    vmax = float(v.max()) if v.notna().any() else 0.0
                    unit = "s"
                    if vmax > 1e15:
                        unit = "ns"
                    elif vmax > 1e12:
                        unit = "ms"
                    elif vmax > 1e9:
                        unit = "s"
                    metric_df["__ts"] = pd.to_datetime(
                        v, unit=unit, errors="coerce", utc=True
                    )
                else:
                    metric_df["__ts"] = pd.to_datetime(
                        ts_raw, errors="coerce", utc=True
                    )
            else:
                metric_df["__ts"] = pd.NaT
            node_feats_metric: List[List[float]] = []
            node_feats_metric_std: List[List[float]] = []

            def _safe_mean(df: pd.DataFrame, col: str, transform=None) -> float:
                if col not in df.columns or df.empty:
                    return 0.0
                series = pd.to_numeric(df[col], errors="coerce").astype(float)
                if transform is not None:
                    series = transform(series)
                val = float(series.mean())
                if np.isnan(val) or np.isinf(val):
                    return 0.0
                return val

            def _safe_std(df: pd.DataFrame, col: str, transform=None) -> float:
                if col not in df.columns or df.empty:
                    return 0.0
                series = pd.to_numeric(df[col], errors="coerce").astype(float)
                if transform is not None:
                    series = transform(series)
                val = float(series.std())
                if np.isnan(val) or np.isinf(val):
                    return 0.0
                return val

            def _safe_max(df: pd.DataFrame, col: str, transform=None) -> float:
                if col not in df.columns or df.empty:
                    return 0.0
                series = pd.to_numeric(df[col], errors="coerce").astype(float)
                if transform is not None:
                    series = transform(series)
                val = float(series.max())
                if np.isnan(val) or np.isinf(val):
                    return 0.0
                return val

            def _safe_series(df: pd.DataFrame, col: str, transform=None) -> pd.Series:
                if col not in df.columns or df.empty:
                    return pd.Series(dtype="float32")
                series = pd.to_numeric(df[col], errors="coerce").astype(float)
                if transform is not None:
                    series = transform(series)
                return series

            for svc in service_list:
                if "service" in metric_df.columns:
                    svc_metrics = metric_df[metric_df["service"].astype(str) == svc]
                elif "service_name" in metric_df.columns:
                    svc_metrics = metric_df[
                        metric_df["service_name"].astype(str) == svc
                    ]
                elif "pod" in metric_df.columns:
                    mask = (
                        metric_df["pod"]
                        .astype(str)
                        .str.contains(svc, case=False, na=False)
                    )
                    svc_metrics = metric_df[mask]
                elif "PodName" in metric_df.columns:
                    mask = (
                        metric_df["PodName"]
                        .astype(str)
                        .str.contains(svc, case=False, na=False)
                    )
                    svc_metrics = metric_df[mask]
                else:
                    svc_metrics = pd.DataFrame()
                if svc_metrics.empty:
                    node_feats_metric.append(
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ]
                    )
                    node_feats_metric_std.append([0.0] * 8)
                    continue
                cpu_mean = 0.0
                cpu_std = 0.0
                for col in ["node_cpu_usage_rate", "container_cpu_usage_seconds_total"]:
                    if col in svc_metrics.columns:
                        transform = lambda s: (
                            s * 100.0
                            if col == "container_cpu_usage_seconds_total"
                            else s
                        )
                        cpu_mean = _safe_mean(svc_metrics, col, transform=transform)
                        cpu_std = _safe_std(svc_metrics, col, transform=transform)
                        break
                mem_mean = 0.0
                mem_std = 0.0
                for col in ["node_memory_usage_bytes", "container_memory_usage_bytes"]:
                    if col in svc_metrics.columns:
                        transform = lambda s: s / (1024.0 * 1024.0)
                        mem_mean = _safe_mean(
                            svc_metrics,
                            col,
                            transform=transform,
                        )
                        mem_std = _safe_std(svc_metrics, col, transform=transform)
                        break
                net_in = 0.0
                net_out = 0.0
                net_in_std = 0.0
                net_out_std = 0.0
                for col in [
                    "container_network_receive_bytes_total",
                    "node_network_receive_bytes_total",
                ]:
                    if col in svc_metrics.columns:
                        net_in = _safe_mean(svc_metrics, col)
                        net_in_std = _safe_std(svc_metrics, col)
                        break
                for col in [
                    "container_network_transmit_bytes_total",
                    "node_network_transmit_bytes_total",
                ]:
                    if col in svc_metrics.columns:
                        net_out = _safe_mean(svc_metrics, col)
                        net_out_std = _safe_std(svc_metrics, col)
                        break
                disk_io = 0.0
                disk_io_std = 0.0
                read_cols = [
                    "container_fs_reads_bytes_total",
                    "node_disk_read_bytes_total",
                ]
                write_cols = [
                    "container_fs_writes_bytes_total",
                    "node_disk_written_bytes_total",
                ]
                read_val = 0.0
                write_val = 0.0
                read_series = None
                write_series = None
                for col in read_cols:
                    if col in svc_metrics.columns:
                        read_val = _safe_mean(svc_metrics, col)
                        read_series = _safe_series(svc_metrics, col)
                        break
                for col in write_cols:
                    if col in svc_metrics.columns:
                        write_val = _safe_mean(svc_metrics, col)
                        write_series = _safe_series(svc_metrics, col)
                        break
                disk_io = read_val + write_val
                if read_series is not None or write_series is not None:
                    if read_series is None:
                        read_series = pd.Series(0.0, index=svc_metrics.index)
                    if write_series is None:
                        write_series = pd.Series(0.0, index=svc_metrics.index)
                    disk_series = read_series + write_series
                    val = float(disk_series.std())
                    if not (np.isnan(val) or np.isinf(val)):
                        disk_io_std = val
                qps = 0.0
                qps_std = 0.0
                for col in [
                    "pod_http_requests_total",
                    "istio_requests_total",
                    "http_requests_total",
                    "request",
                    "response",
                ]:
                    if col in svc_metrics.columns:
                        qps = _safe_mean(svc_metrics, col)
                        qps_std = _safe_std(svc_metrics, col)
                        break
                latency = 0.0
                latency_std = 0.0
                if "pod_http_request_duration_seconds" in svc_metrics.columns:
                    latency = _safe_mean(
                        svc_metrics,
                        "pod_http_request_duration_seconds",
                        transform=lambda s: s * 1000.0,
                    )
                    latency_std = _safe_std(
                        svc_metrics,
                        "pod_http_request_duration_seconds",
                        transform=lambda s: s * 1000.0,
                    )
                else:
                    for col in [
                        "istio_request_duration_milliseconds_p90",
                        "istio_request_duration_milliseconds_p99",
                        "http_request_duration_p90",
                        "http_request_duration_p99",
                        "rrt",
                        "rrt_max",
                    ]:
                        if col in svc_metrics.columns:
                            latency = _safe_mean(svc_metrics, col)
                            latency_std = _safe_std(svc_metrics, col)
                            break
                error_rate = 0.0
                error_rate_std = 0.0
                if "PodSuccessRate(%)" in svc_metrics.columns:
                    succ_series = _safe_series(svc_metrics, "PodSuccessRate(%)")
                    succ = float(succ_series.mean()) if len(succ_series) > 0 else 0.0
                    error_rate = max(0.0, min(1.0, 1.0 - succ / 100.0))
                    if len(succ_series) > 0:
                        er_series = (1.0 - succ_series / 100.0).clip(0.0, 1.0)
                        val = float(er_series.std())
                        if not (np.isnan(val) or np.isinf(val)):
                            error_rate_std = val
                else:
                    ratio_cols = [
                        "error_ratio",
                        "client_error_ratio",
                        "server_error_ratio",
                    ]
                    for col in ratio_cols:
                        if col in svc_metrics.columns:
                            ratio = _safe_series(svc_metrics, col)
                            if len(ratio) > 0:
                                ratio = ratio.clip(lower=0.0)
                                if float(ratio.max()) > 1.0:
                                    ratio = ratio / 100.0
                                error_rate = float(ratio.mean())
                                error_rate_std = float(ratio.std())
                                if np.isnan(error_rate) or np.isinf(error_rate):
                                    error_rate = 0.0
                                if np.isnan(error_rate_std) or np.isinf(error_rate_std):
                                    error_rate_std = 0.0
                            break
                    err_cols = [
                        "istio_request_errors",
                        "http_request_errors",
                        "error",
                        "client_error",
                        "server_error",
                        "timeout",
                    ]
                    tot_cols = [
                        "istio_requests_total",
                        "http_requests_total",
                        "request",
                        "response",
                    ]
                    err_val = 0.0
                    tot_val = 0.0
                    err_s = None
                    tot_s = None
                    for col in err_cols:
                        if col in svc_metrics.columns:
                            err_val = _safe_mean(svc_metrics, col)
                            err_s = _safe_series(svc_metrics, col)
                            break
                    for col in tot_cols:
                        if col in svc_metrics.columns:
                            tot_val = _safe_mean(svc_metrics, col)
                            tot_s = _safe_series(svc_metrics, col)
                            break
                    if error_rate <= 0.0 and tot_val > 0:
                        error_rate = max(0.0, min(1.0, err_val / tot_val))
                    if (
                        error_rate_std <= 0.0
                        and err_s is not None
                        and tot_s is not None
                    ):
                        align = pd.DataFrame({"err": err_s, "tot": tot_s}).dropna()
                        align = align[align["tot"] > 0]
                        if not align.empty:
                            er_series = (align["err"] / align["tot"]).clip(0.0, 1.0)
                            val = float(er_series.std())
                            if not (np.isnan(val) or np.isinf(val)):
                                error_rate_std = val
                node_feats_metric.append(
                    [
                        cpu_mean,
                        mem_mean,
                        net_in,
                        net_out,
                        disk_io,
                        qps,
                        latency,
                        error_rate,
                    ]
                )
                node_feats_metric_std.append(
                    [
                        cpu_std,
                        mem_std,
                        net_in_std,
                        net_out_std,
                        disk_io_std,
                        qps_std,
                        latency_std,
                        error_rate_std,
                    ]
                )
            day_baseline_mean_by_service = {
                svc: np.array(node_feats_metric[i], dtype=np.float32)
                for i, svc in enumerate(service_list)
            }
            day_baseline_std_by_service = {
                svc: np.array(node_feats_metric_std[i], dtype=np.float32)
                for i, svc in enumerate(service_list)
            }
            edges = []
            edge_counts = []
            for (p, c), cnt in call_counts.items():
                if p in service_to_id and c in service_to_id:
                    edges.append((service_to_id[p], service_to_id[c]))
                    edge_counts.append(float(cnt))
            if edges:
                src, dst = zip(*edges)
                edge_dict = {("pod", "calls", "pod"): (list(src), list(dst))}
                num_nodes_dict = {"pod": len(service_list)}
                g = dgl.heterograph(edge_dict, num_nodes_dict=num_nodes_dict)
            else:
                edge_dict = {("pod", "calls", "pod"): ([], [])}
                num_nodes_dict = {"pod": len(service_list)}
                g = dgl.heterograph(edge_dict, num_nodes_dict=num_nodes_dict)
                edge_counts = []
            in_deg = g.in_degrees(etype=("pod", "calls", "pod")).float()
            out_deg = g.out_degrees(etype=("pod", "calls", "pod")).float()
            metric_feat = th.tensor(node_feats_metric, dtype=th.float32)
            feat = th.cat(
                [
                    in_deg.unsqueeze(-1),
                    out_deg.unsqueeze(-1),
                    metric_feat,
                ],
                dim=-1,
            )
            g.ndata["feat"] = feat
            if len(edge_counts) > 0:
                edge_feat_tensor = th.tensor(edge_counts, dtype=th.float32).unsqueeze(1)
                g.edata["feat"] = edge_feat_tensor
            else:
                g.edata["feat"] = th.zeros((0, 1), dtype=th.float32)
            self.graphs.append(g)
            failure_type_label = "network_delay_with_propagation"
            cmdb_id_label = "frontend"
            ts_label = int(datetime.fromisoformat(f"{date_iso}T00:00:00").timestamp())
            gt_set = set()
            if gt_csv.exists():
                df_gt = pd.read_csv(gt_csv)
                if "failure_type" in df_gt.columns and len(df_gt) > 0:
                    ft_series = df_gt["failure_type"].astype(str).str.lower()
                    mask = ~ft_series.apply(
                        lambda s: any(k in s for k in _exclude_failure_type_keywords)
                    )
                    df_gt = df_gt[mask].reset_index(drop=True)
                if len(df_gt) > 0 and "failure_type" in df_gt.columns:
                    failure_type_label = str(
                        df_gt["failure_type"].mode().iloc[0]
                    ).strip()
                    if pd.isna(failure_type_label) or not failure_type_label:
                        failure_type_label = "network_delay_with_propagation"
                if len(df_gt) > 0 and "cmdb_id" in df_gt.columns:
                    cmdb_id_label = str(df_gt["cmdb_id"].iloc[0]).strip() or "frontend"
                if len(df_gt) > 0 and "timestamp" in df_gt.columns:
                    try:
                        ts_label = int(df_gt["timestamp"].iloc[0])
                    except Exception:
                        pass
                for _, row in df_gt.iterrows():
                    ft_row = (
                        str(row.get("failure_type", "")).strip().lower()
                        if "failure_type" in row.index
                        else ""
                    )
                    if ft_row and any(
                        k in ft_row for k in _exclude_failure_type_keywords
                    ):
                        continue
                    svc = str(row["cmdb_id"]).strip().lower()
                    row_gt = set()
                    for s_name, nid in service_to_id.items():
                        s_low = s_name.lower()
                        if svc == s_low or svc in s_low or s_low in svc:
                            gt_set.add(("pod", nid))
                            row_gt.add(("pod", nid))
                    if row_gt:
                        ft = (
                            str(row.get("failure_type", "")).strip()
                            if "failure_type" in row.index
                            else failure_type_label
                        )
                        ts_row = ts_label
                        if "timestamp" in row.index:
                            try:
                                ts_row = int(row["timestamp"])
                            except Exception:
                                pass
                        t_anchor = pd.to_datetime(ts_row, unit="s", utc=True)
                        window_secs_list = [3 * 60, 5 * 60, 10 * 60, 20 * 60]
                        lag_secs_list = [0, 5 * 60, 10 * 60]
                        event_feat = np.zeros((len(service_list), 8), dtype=np.float32)

                        def _select_svc_df(
                            df: pd.DataFrame, svc_name: str
                        ) -> pd.DataFrame:
                            if df.empty:
                                return df
                            if "service" in df.columns:
                                return df[df["service"].astype(str) == svc_name]
                            if "service_name" in df.columns:
                                return df[df["service_name"].astype(str) == svc_name]
                            if "pod" in df.columns:
                                mask = (
                                    df["pod"]
                                    .astype(str)
                                    .str.contains(svc_name, case=False, na=False)
                                )
                                return df[mask]
                            if "PodName" in df.columns:
                                mask = (
                                    df["PodName"]
                                    .astype(str)
                                    .str.contains(svc_name, case=False, na=False)
                                )
                                return df[mask]
                            return pd.DataFrame()

                        if (
                            "__ts" in metric_df.columns
                            and metric_df["__ts"].notna().any()
                        ):
                            for s_idx, svc_name in enumerate(service_list):
                                mean_vec = day_baseline_mean_by_service.get(
                                    svc_name, np.zeros(8, dtype=np.float32)
                                )
                                std_vec = day_baseline_std_by_service.get(
                                    svc_name, np.zeros(8, dtype=np.float32)
                                )
                                std_vec = np.where(std_vec <= 1e-6, 1.0, std_vec)
                                best = np.zeros(8, dtype=np.float32)
                                for lag_s in lag_secs_list:
                                    t_end = t_anchor - pd.Timedelta(seconds=lag_s)
                                    for w_s in window_secs_list:
                                        t_start = t_end - pd.Timedelta(seconds=w_s)
                                        df_win = metric_df[
                                            (metric_df["__ts"] >= t_start)
                                            & (metric_df["__ts"] <= t_end)
                                        ]
                                        svc_win = _select_svc_df(df_win, svc_name)
                                        if svc_win.empty:
                                            continue
                                        cpu_peak = 0.0
                                        for col in [
                                            "node_cpu_usage_rate",
                                            "container_cpu_usage_seconds_total",
                                        ]:
                                            if col in svc_win.columns:
                                                transform = lambda s: (
                                                    s * 100.0
                                                    if col
                                                    == "container_cpu_usage_seconds_total"
                                                    else s
                                                )
                                                cpu_peak = _safe_max(
                                                    svc_win, col, transform=transform
                                                )
                                                break
                                        mem_peak = 0.0
                                        for col in [
                                            "node_memory_usage_bytes",
                                            "container_memory_usage_bytes",
                                        ]:
                                            if col in svc_win.columns:
                                                mem_peak = _safe_max(
                                                    svc_win,
                                                    col,
                                                    transform=(
                                                        lambda s: s / (1024.0 * 1024.0)
                                                    ),
                                                )
                                                break
                                        net_in_peak = 0.0
                                        net_out_peak = 0.0
                                        for col in [
                                            "container_network_receive_bytes_total",
                                            "node_network_receive_bytes_total",
                                        ]:
                                            if col in svc_win.columns:
                                                net_in_peak = _safe_max(svc_win, col)
                                                break
                                        for col in [
                                            "container_network_transmit_bytes_total",
                                            "node_network_transmit_bytes_total",
                                        ]:
                                            if col in svc_win.columns:
                                                net_out_peak = _safe_max(svc_win, col)
                                                break
                                        read_s = None
                                        write_s = None
                                        for col in [
                                            "container_fs_reads_bytes_total",
                                            "node_disk_read_bytes_total",
                                        ]:
                                            if col in svc_win.columns:
                                                read_s = _safe_series(svc_win, col)
                                                break
                                        for col in [
                                            "container_fs_writes_bytes_total",
                                            "node_disk_written_bytes_total",
                                        ]:
                                            if col in svc_win.columns:
                                                write_s = _safe_series(svc_win, col)
                                                break
                                        disk_peak = 0.0
                                        if read_s is not None or write_s is not None:
                                            if read_s is None:
                                                read_s = pd.Series(
                                                    0.0, index=svc_win.index
                                                )
                                            if write_s is None:
                                                write_s = pd.Series(
                                                    0.0, index=svc_win.index
                                                )
                                            disk_peak = float((read_s + write_s).max())
                                            if np.isnan(disk_peak) or np.isinf(
                                                disk_peak
                                            ):
                                                disk_peak = 0.0
                                        qps_peak = 0.0
                                        for col in [
                                            "pod_http_requests_total",
                                            "istio_requests_total",
                                            "http_requests_total",
                                            "request",
                                            "response",
                                        ]:
                                            if col in svc_win.columns:
                                                qps_peak = _safe_max(svc_win, col)
                                                break
                                        lat_peak = 0.0
                                        if (
                                            "pod_http_request_duration_seconds"
                                            in svc_win.columns
                                        ):
                                            lat_peak = _safe_max(
                                                svc_win,
                                                "pod_http_request_duration_seconds",
                                                transform=(lambda s: s * 1000.0),
                                            )
                                        else:
                                            for col in [
                                                "istio_request_duration_milliseconds_p90",
                                                "istio_request_duration_milliseconds_p99",
                                                "http_request_duration_p90",
                                                "http_request_duration_p99",
                                                "rrt",
                                                "rrt_max",
                                            ]:
                                                if col in svc_win.columns:
                                                    lat_peak = _safe_max(svc_win, col)
                                                    break
                                        err_peak = 0.0
                                        if "PodSuccessRate(%)" in svc_win.columns:
                                            succ_s = _safe_series(
                                                svc_win, "PodSuccessRate(%)"
                                            )
                                            if len(succ_s) > 0:
                                                er_s = (1.0 - succ_s / 100.0).clip(
                                                    0.0, 1.0
                                                )
                                                err_peak = float(er_s.max())
                                            if np.isnan(err_peak) or np.isinf(err_peak):
                                                err_peak = 0.0
                                        else:
                                            err_s2 = None
                                            tot_s2 = None
                                            for col in [
                                                "error_ratio",
                                                "client_error_ratio",
                                                "server_error_ratio",
                                            ]:
                                                if col in svc_win.columns:
                                                    ratio = _safe_series(svc_win, col)
                                                    if len(ratio) > 0:
                                                        ratio = ratio.clip(lower=0.0)
                                                        if float(ratio.max()) > 1.0:
                                                            ratio = ratio / 100.0
                                                        err_peak = float(ratio.max())
                                                        if np.isnan(
                                                            err_peak
                                                        ) or np.isinf(err_peak):
                                                            err_peak = 0.0
                                                    break
                                            for col in [
                                                "istio_request_errors",
                                                "http_request_errors",
                                                "error",
                                                "client_error",
                                                "server_error",
                                                "timeout",
                                            ]:
                                                if col in svc_win.columns:
                                                    err_s2 = _safe_series(svc_win, col)
                                                    break
                                            for col in [
                                                "istio_requests_total",
                                                "http_requests_total",
                                                "request",
                                                "response",
                                            ]:
                                                if col in svc_win.columns:
                                                    tot_s2 = _safe_series(svc_win, col)
                                                    break
                                            if (
                                                err_peak <= 0.0
                                                and err_s2 is not None
                                                and tot_s2 is not None
                                            ):
                                                align = pd.DataFrame(
                                                    {"err": err_s2, "tot": tot_s2}
                                                ).dropna()
                                                align = align[align["tot"] > 0]
                                                if not align.empty:
                                                    ratio = (
                                                        align["err"]
                                                        / (align["tot"] + 1e-6)
                                                    ).clip(0.0, 1.0)
                                                    err_peak = float(ratio.max())
                                                    if np.isnan(err_peak) or np.isinf(
                                                        err_peak
                                                    ):
                                                        err_peak = 0.0
                                        peak_vec = np.array(
                                            [
                                                cpu_peak,
                                                mem_peak,
                                                net_in_peak,
                                                net_out_peak,
                                                disk_peak,
                                                qps_peak,
                                                lat_peak,
                                                err_peak,
                                            ],
                                            dtype=np.float32,
                                        )
                                        z = np.abs(
                                            (peak_vec - mean_vec) / (std_vec + 1e-6)
                                        )
                                        best = np.maximum(best, z.astype(np.float32))
                                event_feat[s_idx, :] = best
                        self.per_row_node_feats.append(th.from_numpy(event_feat))
                        self.per_row_labels.append(
                            {
                                "timestamp": ts_row,
                                "level": "service",
                                "cmdb_id": str(row.get("cmdb_id", "")).strip()
                                or cmdb_id_label,
                                "failure_type": ft or failure_type_label,
                            }
                        )
                        self.per_row_groundtruths.append(row_gt)
                        self.per_row_graph_indices.append(len(self.graphs) - 1)
            if not gt_set:
                logger.warning(f"{date} Groundtruth node not found")
            self.labels.append(
                {
                    "timestamp": ts_label,
                    "level": "service",
                    "cmdb_id": cmdb_id_label,
                    "failure_type": failure_type_label,
                }
            )
            self.groundtruths.append(gt_set)
        logger.info(f"DatasetAIOps2025 Build complete: {len(self.graphs)} Sample image")
        if len(self.graphs) > 0:
            self._init_nan_nodes_and_edges()
        else:
            self.nan_nodes = {}
            self.nan_edges = {}
            for ntype in self.ntypes:
                self.nan_nodes[ntype] = th.tensor([])
            for etype in self.etypes:
                self.nan_edges[etype] = th.tensor([])

    def _init_nan_nodes_and_edges(self):
        self.nan_nodes = {}
        self.nan_edges = {}
        ntype = self.ntypes[0]
        nans = []
        for g in self.graphs:
            data = (
                g.ndata["feat"]
                if isinstance(g.ndata["feat"], th.Tensor)
                else g.ndata["feat"][ntype]
            )
            nan = th.where(th.isnan(data).all(dim=1), th.tensor(True), th.tensor(False))
            if len(g.ntypes) == 1:
                g.ndata["nan"] = nan.reshape(-1, 1)
            else:
                g.ndata["nan"] = {ntype: nan.reshape(-1, 1)}
            nans.append(nan)
        self.nan_nodes[ntype] = th.stack(nans, dim=0)
        etype = self.etypes[0]
        nans = []
        for g in self.graphs:
            data = (
                g.edata["feat"]
                if isinstance(g.edata["feat"], th.Tensor)
                else g.edata["feat"][etype]
            )
            nan = th.where(th.isnan(data).all(dim=1), th.tensor(True), th.tensor(False))
            if len(g.canonical_etypes) == 1:
                g.edata["nan"] = nan.reshape(-1, 1)
            else:
                g.edata["nan"] = {etype: nan.reshape(-1, 1)}
            nans.append(nan)
        self.nan_edges[etype] = th.stack(nans, dim=0)

    def get_nan_nodes(self):
        if hasattr(self, "nan_nodes") and self.nan_nodes:
            return self.nan_nodes
        self._init_nan_nodes_and_edges()
        return self.nan_nodes

    def get_nan_edges(self):
        if hasattr(self, "nan_edges") and self.nan_edges:
            return self.nan_edges
        self._init_nan_nodes_and_edges()
        return self.nan_edges

    def get_stacked_nfeat(self):
        feats = {}
        ntype = self.ntypes[0]
        feats[ntype] = []
        for g in self.graphs:
            feat = (
                g.ndata["feat"]
                if isinstance(g.ndata["feat"], th.Tensor)
                else g.ndata["feat"][ntype]
            )
            feats[ntype].append(feat)
        feats[ntype] = th.stack(feats[ntype], dim=0)
        return feats

    def get_stacked_efeat(self):
        feats = {}
        etype = self.etypes[0]
        feats[etype] = []
        for g in self.graphs:
            feat = (
                g.edata["feat"]
                if isinstance(g.edata["feat"], th.Tensor)
                else g.edata["feat"][etype]
            )
            feats[etype].append(feat)
        feats[etype] = th.stack(feats[etype], dim=0)
        return feats


{
    "cells": [],
    "metadata": {"language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 2,
}
