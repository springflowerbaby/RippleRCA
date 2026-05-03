"""Minimal trace and metric utilities used by RippleRCA."""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]
import torch as th  # pyright: ignore[reportMissingImports]


def _canonicalize_service_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    value = str(name).strip().strip('"').strip("'")
    if not value:
        return None
    lower = value.lower()
    for prefix in ("http://", "https://"):
        if lower.startswith(prefix):
            value = value[len(prefix) :]
            break
    value = value.split("?", 1)[0].split("#", 1)[0]
    value = value.split("/", 1)[0]
    value = value.split(":", 1)[0]
    for suffix in (".svc.cluster.local", ".cluster.local", ".svc"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    parts = value.split("-")
    if len(parts) >= 3 and parts[-1].isalnum() and 4 <= len(parts[-1]) <= 10:
        if any(ch.isdigit() for ch in parts[-2]) or len(parts[-2]) >= 6:
            value = "-".join(parts[:-2])
    return value.lower().strip() or None


class TraceParser:
    def __init__(
        self, trace_dir: str, min_call_frequency: int = 5, time_window: int = 60
    ) -> None:
        self.trace_dir = Path(trace_dir)
        self.min_call_frequency = min_call_frequency
        self.time_window = time_window

    def _load_csv_traces(self) -> pd.DataFrame:
        csv_files = sorted(self.trace_dir.glob("*.csv"))
        if not csv_files:
            return pd.DataFrame()
        frames = [pd.read_csv(path) for path in csv_files]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _load_json_traces(self) -> pd.DataFrame:
        json_files = sorted(self.trace_dir.glob("*.json"))
        rows: List[Dict[str, object]] = []
        for path in json_files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            spans = (
                payload.get("data", payload) if isinstance(payload, dict) else payload
            )
            if not isinstance(spans, list):
                continue
            for item in spans:
                if not isinstance(item, dict):
                    continue
                rows.append(item)
        return pd.DataFrame(rows)

    def parse_traces(
        self,
        start_time: Optional[pd.Timestamp] = None,
        end_time: Optional[pd.Timestamp] = None,
    ) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], int]]:
        trace_df_raw = self._load_csv_traces()
        if trace_df_raw.empty:
            trace_df_raw = self._load_json_traces()
        if trace_df_raw.empty:
            return pd.DataFrame(), {}
        if "StartTimeUnixNano" in trace_df_raw.columns:
            timestamp = pd.to_datetime(
                trace_df_raw["StartTimeUnixNano"], unit="ns", errors="coerce"
            )
        elif "startTime" in trace_df_raw.columns:
            timestamp = pd.to_datetime(
                trace_df_raw["startTime"], unit="us", errors="coerce"
            )
        else:
            timestamp = pd.to_datetime(trace_df_raw.iloc[:, 0], errors="coerce")
        span_to_pod = {}
        if {"SpanID", "PodName"}.issubset(trace_df_raw.columns):
            span_to_pod = dict(zip(trace_df_raw["SpanID"], trace_df_raw["PodName"]))
        records: List[Dict[str, object]] = []
        for _, row in trace_df_raw.iterrows():
            child_service = _canonicalize_service_name(
                row.get("PodName") or row.get("serviceName")
            )
            parent_service = None
            parent_id = row.get("ParentID") or row.get("parentSpanId")
            if isinstance(parent_id, str) and parent_id and parent_id != "root":
                parent_service = _canonicalize_service_name(span_to_pod.get(parent_id))
            if child_service is None:
                continue
            duration = float(row.get("Duration", row.get("duration", 0.0)) or 0.0)
            records.append(
                {
                    "timestamp": (
                        timestamp.iloc[len(records)]
                        if len(records) < len(timestamp)
                        else pd.NaT
                    ),
                    "parent_service": parent_service,
                    "child_service": child_service,
                    "duration": duration,
                }
            )
        trace_df = pd.DataFrame(records).dropna(subset=["timestamp"])
        if start_time is not None:
            trace_df = trace_df[trace_df["timestamp"] >= start_time]
        if end_time is not None:
            trace_df = trace_df[trace_df["timestamp"] <= end_time]
        call_counts: Dict[Tuple[str, str], int] = {}
        for _, row in trace_df.iterrows():
            parent = row.get("parent_service")
            child = row.get("child_service")
            if parent and child:
                edge = (str(parent), str(child))
                call_counts[edge] = call_counts.get(edge, 0) + 1
        call_counts = {
            edge: count
            for edge, count in call_counts.items()
            if count >= self.min_call_frequency
        }
        return trace_df.reset_index(drop=True), call_counts

    def get_trace_topology(
        self,
        call_counts: Dict[Tuple[str, str], int],
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        services = sorted({svc for edge in call_counts for svc in edge})
        edges = sorted(call_counts.keys())
        return services, edges


class MetricAggregator:
    def __init__(self, metric_dir: str, sampling_interval: int = 10) -> None:
        self.metric_dir = Path(metric_dir)
        self.sampling_interval = sampling_interval

    def load_prometheus_metrics(
        self,
        start_time: Optional[pd.Timestamp] = None,
        end_time: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        csv_files = sorted(self.metric_dir.glob("*.csv"))
        if not csv_files:
            return pd.DataFrame()
        frames = [pd.read_csv(path) for path in csv_files]
        metric_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if metric_df.empty:
            return metric_df
        for candidate in ("timestamp", "time", "TimeStamp", "__time"):
            if candidate in metric_df.columns:
                metric_df["timestamp"] = pd.to_datetime(
                    metric_df[candidate], errors="coerce"
                )
                break
        if "timestamp" not in metric_df.columns:
            metric_df["timestamp"] = pd.to_datetime(
                metric_df.iloc[:, 0], errors="coerce"
            )
        if start_time is not None:
            metric_df = metric_df[metric_df["timestamp"] >= start_time]
        if end_time is not None:
            metric_df = metric_df[metric_df["timestamp"] <= end_time]
        return metric_df.reset_index(drop=True)


def _infer_metric_columns(metric_df: pd.DataFrame) -> List[str]:
    ignore = {
        "timestamp",
        "time",
        "TimeStamp",
        "__time",
        "service",
        "service_name",
        "svc",
        "pod",
        "pod_name",
        "instance",
    }
    numeric_cols = []
    for column in metric_df.columns:
        if column in ignore:
            continue
        if pd.api.types.is_numeric_dtype(metric_df[column]):
            numeric_cols.append(column)
    return numeric_cols[:5]


def build_service_timeseries(
    trace_parser: TraceParser,
    metric_aggregator: MetricAggregator,
    trace_df: pd.DataFrame,
    call_counts: Dict[Tuple[str, str], int],
    metric_df: pd.DataFrame,
    tau_max: int,
    time_window: int = 60,
) -> Tuple[th.Tensor, List[str], Dict[str, th.Tensor]]:
    service_list, edge_list = trace_parser.get_trace_topology(call_counts)
    if metric_df.empty and trace_df.empty:
        raise ValueError(
            "No trace or metric records were found for service-level aggregation."
        )
    metric_services = set()
    for column in ("service", "service_name", "svc", "pod", "pod_name", "instance"):
        if column in metric_df.columns:
            metric_services.update(
                filter(
                    None,
                    (
                        _canonicalize_service_name(value)
                        for value in metric_df[column].dropna().tolist()
                    ),
                )
            )
    service_list = sorted(set(service_list) | metric_services)
    if not service_list:
        raise ValueError("No services were found while building service-level inputs.")
    all_times = []
    if not trace_df.empty:
        all_times.extend(trace_df["timestamp"].dropna().tolist())
    if not metric_df.empty:
        all_times.extend(metric_df["timestamp"].dropna().tolist())
    time_axis = pd.to_datetime(sorted(set(all_times)))
    if len(time_axis) == 0:
        raise ValueError(
            "No timestamps were found while building service-level inputs."
        )
    trace_metric_names = ["trace_count", "trace_duration", "trace_error"]
    metric_cols = _infer_metric_columns(metric_df)
    total_features = len(trace_metric_names) + len(metric_cols)
    service_index = {service: idx for idx, service in enumerate(service_list)}
    time_index = {ts: idx for idx, ts in enumerate(time_axis)}
    x_tensor = th.zeros(
        (len(time_axis), len(service_list), total_features), dtype=th.float32
    )
    if not trace_df.empty:
        grouped = trace_df.groupby(["timestamp", "child_service"], dropna=True)
        for (ts, service), group in grouped:
            service_name = _canonicalize_service_name(service)
            if service_name not in service_index or ts not in time_index:
                continue
            ti = time_index[ts]
            si = service_index[service_name]
            x_tensor[ti, si, 0] = float(len(group))
            x_tensor[ti, si, 1] = (
                float(group["duration"].mean()) if "duration" in group else 0.0
            )
            x_tensor[ti, si, 2] = float(
                (group.get("duration", pd.Series(dtype=float)) <= 0).sum()
            )
    if not metric_df.empty and metric_cols:
        service_col = None
        for candidate in (
            "service",
            "service_name",
            "svc",
            "pod",
            "pod_name",
            "instance",
        ):
            if candidate in metric_df.columns:
                service_col = candidate
                break
        if service_col is not None:
            grouped = metric_df.groupby(["timestamp", service_col], dropna=True)
            for (ts, service), group in grouped:
                service_name = _canonicalize_service_name(service)
                if service_name not in service_index or ts not in time_index:
                    continue
                ti = time_index[ts]
                si = service_index[service_name]
                for offset, column in enumerate(metric_cols):
                    x_tensor[ti, si, len(trace_metric_names) + offset] = float(
                        group[column].mean()
                    )
    baseline_steps = max(1, min(len(time_axis), tau_max + 1))
    x_norm: Dict[str, th.Tensor] = {}
    baseline = x_tensor[:baseline_steps].mean(dim=0)
    for service, idx in service_index.items():
        x_norm[service] = baseline[idx]
    return x_tensor, service_list, x_norm
