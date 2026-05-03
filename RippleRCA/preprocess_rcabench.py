from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]

FEATURE_KEYS: Tuple[str, ...] = (
    "cpu",
    "mem",
    "diskio",
    "net_in",
    "net_out",
    "workload",
    "latency",
    "error",
)
RCABENCH_FAULT_TYPE_MAP: Dict[int, str] = {
    0: "pod-failure",
    1: "pod-failure",
    2: "container-kill",
    5: "request-abort",
    6: "response-abort",
    7: "request-delay",
    8: "response-delay",
    9: "replace-body",
    10: "replace-body",
    11: "replace-path",
    12: "replace-method",
    13: "replace-code",
    15: "partition",
    16: "time",
    17: "request-delay",
    18: "loss",
    20: "corrupt",
    21: "bandwidth",
    22: "partition",
    23: "request-delay",
    24: "return",
    25: "exception",
    27: "stress",
    28: "stress",
    29: "mysql",
}
RCABENCH_FAILURE_NAME_KEYS: Tuple[Tuple[str, str], ...] = (
    ("container-kill", "container-kill"),
    ("pod-failure", "pod-failure"),
    ("pod-kill", "pod-failure"),
    ("request-abort", "request-abort"),
    ("response-abort", "response-abort"),
    ("request-delay", "request-delay"),
    ("response-delay", "response-delay"),
    ("replace-method", "replace-method"),
    ("replace-path", "replace-path"),
    ("replace-body", "replace-body"),
    ("replace-code", "replace-code"),
    ("patch-body", "replace-body"),
    ("bandwidth", "bandwidth"),
    ("loss", "loss"),
    ("partition", "partition"),
    ("corrupt", "corrupt"),
    ("exception", "exception"),
    ("stress", "stress"),
    ("latency", "request-delay"),
    ("delay", "request-delay"),
    ("dns", "partition"),
    ("mysql", "mysql"),
    ("time", "time"),
    ("return", "return"),
)


def infer_rcabench_failure_type(scenario_name: str, injection: Dict[str, Any]) -> str:
    name = str(scenario_name or "").lower()
    for key, label in RCABENCH_FAILURE_NAME_KEYS:
        if key in name:
            return label
    fault_type = _safe_int(injection.get("fault_type"))
    if fault_type is not None and fault_type in RCABENCH_FAULT_TYPE_MAP:
        return RCABENCH_FAULT_TYPE_MAP[fault_type]
    display_config = injection.get("display_config")
    try:
        if isinstance(display_config, str):
            display = json.loads(display_config)
        elif isinstance(display_config, dict):
            display = display_config
        else:
            display = {}
    except Exception:
        display = {}
    display_text = json.dumps(display, ensure_ascii=False).lower() if display else ""
    if "latency" in display_text or "jitter" in display_text:
        return "request-delay"
    if "replace_method" in display_text:
        return "replace-method"
    if "patch" in display_text and "body" in display_text:
        return "replace-body"
    if "domain" in display_text and "dns" in name:
        return "partition"
    if "operation_type" in display_text or "db_name" in display_text:
        return "mysql"
    return "unknown"


def canonicalize_service_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    s = str(name).strip().strip('"').strip("'")
    if not s:
        return None
    lower = s.lower()
    for prefix in ("http://", "https://"):
        if lower.startswith(prefix):
            s = s[len(prefix) :]
            break
    s = s.split("?", 1)[0].split("#", 1)[0]
    s = s.split("/", 1)[0]
    s = s.split(":", 1)[0]
    for suf in (".svc.cluster.local", ".cluster.local", ".svc"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    s = s.strip().lower()
    return s or None


def _parse_service_from_span(span_name: str) -> Optional[str]:
    if not isinstance(span_name, str):
        span_name = str(span_name)
    s = span_name.strip()
    if not s:
        return None
    lower = s.lower()
    for prefix in ("http://", "https://"):
        if prefix in lower:
            try:
                part = s[lower.index(prefix) + len(prefix) :]
                host = part.split("/", 1)[0]
                host = host.split(":", 1)[0]
                host = host.strip()
                if host:
                    return host
            except Exception:
                pass
    return None


def _parse_service_from_metric_name(metric: str) -> Optional[str]:
    m = str(metric)
    if not m:
        return None
    for sep in ("__", "::", ":", "_"):
        if sep in m:
            head = m.split(sep, 1)[0]
            if head:
                m = head
                break
    s = m.strip()
    if not s:
        return None
    if "-" in s:
        return s
    return None


def _require_parquet_engine() -> None:
    try:
        import pyarrow  # noqa: F401

        return
    except Exception:
        pass
    try:
        import fastparquet  # noqa: F401

        return
    except Exception:
        pass
    raise RuntimeError(
        "无法读取 parquet：请安装 parquet 引擎。\n"
        "推荐：pip install pyarrow\n"
        "或：pip install fastparquet"
    )


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_int(x) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return None


def _detect_col(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        cand_l = cand.lower()
        if cand_l in cols_l:
            return cols_l[cand_l]
    return None


def _infer_service_from_any(row: pd.Series) -> Optional[str]:
    for key in (
        "service",
        "service_name",
        "cmdb_id",
        "container",
        "container_name",
        "pod",
        "pod_name",
    ):
        if key in row and pd.notna(row[key]):
            return str(row[key])
    return None


def _coerce_epoch_seconds(ts) -> Optional[int]:
    try:
        t = pd.to_datetime(ts, errors="coerce", utc=True)
        if pd.notna(t):
            return int(t.to_pydatetime().timestamp())
    except Exception:
        pass
    v = _safe_int(ts)
    if v is None:
        return None
    if v > 10**15:  # ns
        return int(v / 1_000_000_000)
    if v > 10**12:  # us
        return int(v / 1_000_000)
    if v > 10**10:  # ms
        return int(v / 1_000)
    return v


def _map_metric_to_feature(metric_name: str) -> Optional[str]:
    m = str(metric_name).lower()
    if m in {
        "k8s.pod.phase",
        "k8s.container.ready",
        "k8s.namespace.phase",
    }:
        return "error"
    if m in {
        "k8s.deployment.desired",
        "k8s.deployment.available",
        "k8s.replicaset.desired",
        "k8s.replicaset.available",
        "k8s.statefulset.updated_pods",
        "k8s.statefulset.ready_pods",
        "k8s.statefulset.desired_pods",
        "k8s.statefulset.current_pods",
        "queuesize",
    }:
        return "workload"
    if any(k in m for k in ("cpu", "container_cpu", "process_cpu")):
        return "cpu"
    if any(k in m for k in ("mem", "memory", "rss", "heap")):
        return "mem"
    if any(k in m for k in ("disk", "blkio", "fs_read", "fs_write", "io")):
        return "diskio"
    if any(
        k in m
        for k in ("net_in", "receive_bytes", "rx_bytes", "network_receive", "recv")
    ):
        return "net_in"
    if any(
        k in m
        for k in ("net_out", "transmit_bytes", "tx_bytes", "network_transmit", "send")
    ):
        return "net_out"
    if any(
        k in m
        for k in ("qps", "rps", "throughput", "request_total", "requests", "workload")
    ):
        return "workload"
    if any(k in m for k in ("latency", "duration", "p95", "p90", "p99", "rt")):
        return "latency"
    if any(k in m for k in ("error", "5xx", "exception", "fail", "timeout")):
        return "error"
    return None


def _map_metric_to_feature_reference(metric_name: str) -> Optional[str]:
    m = str(metric_name).lower()
    if any(k in m for k in ("cpu", "container_cpu", "process_cpu")):
        return "cpu"
    if any(k in m for k in ("mem", "memory", "rss", "heap")):
        return "mem"
    if any(k in m for k in ("disk", "blkio", "fs_read", "fs_write", "io")):
        return "diskio"
    if any(
        k in m
        for k in ("net_in", "receive_bytes", "rx_bytes", "network_receive", "recv")
    ):
        return "net_in"
    if any(
        k in m
        for k in ("net_out", "transmit_bytes", "tx_bytes", "network_transmit", "send")
    ):
        return "net_out"
    if any(
        k in m
        for k in ("qps", "rps", "throughput", "request_total", "requests", "workload")
    ):
        return "workload"
    if any(k in m for k in ("latency", "duration", "p95", "p90", "p99", "rt")):
        return "latency"
    if any(k in m for k in ("error", "5xx", "exception", "fail", "timeout")):
        return "error"
    return None


def _build_mapping_debug(
    scenario: str,
    normal_long: pd.DataFrame,
    abnormal_long: pd.DataFrame,
    top_unmapped_n: int = 30,
) -> Dict[str, Any]:
    metrics = (
        pd.concat([normal_long["metric"], abnormal_long["metric"]], axis=0)
        .dropna()
        .astype(str)
    )
    if metrics.empty:
        return {
            "scenario": scenario,
            "rows_total": 0,
            "unique_metrics_total": 0,
            "rows": {},
            "unique_metrics": {},
        }
    df = pd.DataFrame({"metric": metrics})
    df["bucket_ref"] = df["metric"].map(_map_metric_to_feature_reference)
    df["bucket_cur"] = df["metric"].map(_map_metric_to_feature)

    def _pack(sub: pd.DataFrame) -> Dict[str, Any]:
        rows_total = int(len(sub))
        bucket_keys = list(FEATURE_KEYS)
        ref_counts = {k: int((sub["bucket_ref"] == k).sum()) for k in bucket_keys}
        cur_counts = {k: int((sub["bucket_cur"] == k).sum()) for k in bucket_keys}
        delta_counts = {k: int(cur_counts[k] - ref_counts[k]) for k in bucket_keys}
        unmapped_ref = int(sub["bucket_ref"].isna().sum())
        unmapped_cur = int(sub["bucket_cur"].isna().sum())
        unmapped_hits = {
            "reference": unmapped_ref,
            "current": unmapped_cur,
            "delta_current_minus_reference": int(unmapped_cur - unmapped_ref),
        }
        migration = (
            sub.groupby(
                [
                    sub["bucket_ref"].fillna("UNMAPPED_REF"),
                    sub["bucket_cur"].fillna("UNMAPPED_CUR"),
                ]
            )
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        migrations = [
            {"from": str(r.iloc[0]), "to": str(r.iloc[1]), "count": int(r.iloc[2])}
            for _, r in migration.iterrows()
            if int(r.iloc[2]) > 0
        ]
        unmapped_metrics = (
            sub[sub["bucket_cur"].isna()]["metric"]
            .value_counts()
            .head(top_unmapped_n)
            .to_dict()
        )
        return {
            "rows_total": rows_total,
            "bucket_counts_reference": ref_counts,
            "bucket_counts_current": cur_counts,
            "bucket_delta_current_minus_reference": delta_counts,
            "unmapped_hits": unmapped_hits,
            "mapped_migrations": migrations[:100],
            "top_unmapped_metrics_current": [
                {"metric": k, "count": int(v)} for k, v in unmapped_metrics.items()
            ],
        }

    rows_pack = _pack(df)
    unique_df = df.drop_duplicates(subset=["metric"]).copy()
    unique_pack = _pack(unique_df)
    return {
        "scenario": scenario,
        "rows_total": rows_pack["rows_total"],
        "unique_metrics_total": unique_pack["rows_total"],
        "rows": rows_pack,
        "unique_metrics": unique_pack,
    }


def _load_metrics_long(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return pd.DataFrame(columns=["ts", "service", "metric", "value"])
    ts_col = _detect_col(df.columns, ("timestamp", "time", "ts", "datetime", "__ts"))
    metric_col = _detect_col(df.columns, ("metric", "metric_name", "name"))
    value_col = _detect_col(df.columns, ("value", "v", "metric_value"))
    service_col = _detect_col(
        df.columns,
        ("service", "service_name", "container", "container_name", "pod", "pod_name"),
    )
    if ts_col and metric_col and value_col:
        out = df[
            [ts_col, metric_col, value_col] + ([service_col] if service_col else [])
        ].copy()
        out.rename(
            columns={ts_col: "ts_raw", metric_col: "metric", value_col: "value"},
            inplace=True,
        )
        if service_col:
            out.rename(columns={service_col: "service"}, inplace=True)
        else:
            out["service"] = None
        out["ts"] = out["ts_raw"].map(_coerce_epoch_seconds)
        out.drop(columns=["ts_raw"], inplace=True)
        out["service"] = out["service"].astype("string")
        out["metric"] = out["metric"].astype("string")
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna(subset=["ts", "metric", "value"])
        if (
            out["service"].isna().any()
            or (out["service"].astype("string") == "unknown").all()
        ):
            parsed = out["metric"].map(_parse_service_from_metric_name)
            out["service"] = out["service"].fillna(parsed)
            if out["service"].isna().any():
                for alt in (
                    "service",
                    "service_name",
                    "container",
                    "container_name",
                    "pod",
                    "pod_name",
                ):
                    if alt in df.columns:
                        out["service"] = out["service"].fillna(df[alt].astype("string"))
                        break
        out["service"] = out["service"].fillna("unknown")
        return out[["ts", "service", "metric", "value"]]
    if ts_col:
        wide = df.copy()
        wide.rename(columns={ts_col: "ts_raw"}, inplace=True)
        id_vars = ["ts_raw"]
        value_vars = [c for c in wide.columns if c not in id_vars]
        melted = wide.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="metric",
            value_name="value",
        )
        melted["ts"] = melted["ts_raw"].map(_coerce_epoch_seconds)
        melted.drop(columns=["ts_raw"], inplace=True)
        melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
        melted = melted.dropna(subset=["ts", "value"])
        melted["service"] = melted["metric"].map(
            lambda x: _parse_service_from_metric_name(x) or "unknown"
        )
        return melted[["ts", "service", "metric", "value"]]
    raise ValueError(
        f"无法识别 metrics parquet schema: {parquet_path}\n"
        f"columns={list(df.columns)[:50]}\n"
        "期望至少包含 timestamp/time + metric_name + value（long）或 timestamp/time（wide）。"
    )


def _compute_service_timeseries(
    normal_long: pd.DataFrame,
    abnormal_long: pd.DataFrame,
    normal_range: Tuple[int, int],
    abnormal_range: Tuple[int, int],
    freq_s: int,
    max_services: int = 200,
    gt_service: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    n0, n1 = normal_range
    a0, a1 = abnormal_range
    ts_index = list(range(a0 - (a0 % freq_s), a1 + 1, freq_s))

    def add_feature_bucket(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["feature"] = df["metric"].map(_map_metric_to_feature)
        df = df.dropna(subset=["feature"])
        return df

    normal_b = add_feature_bucket(normal_long)
    abnormal_b = add_feature_bucket(abnormal_long)
    normal_b = normal_b[(normal_b["ts"] >= n0) & (normal_b["ts"] <= n1)]
    abnormal_b = abnormal_b[(abnormal_b["ts"] >= a0) & (abnormal_b["ts"] <= a1)]
    services = (
        pd.concat([normal_b["service"], abnormal_b["service"]], axis=0)
        .dropna()
        .astype("string")
        .value_counts()
        .head(max_services)
        .index.tolist()
    )
    if not services:
        services = ["unknown"]
    if gt_service and gt_service not in services:
        services = [gt_service] + services
        services = services[:max_services]
    normal_b = normal_b[normal_b["service"].isin(services)]
    abnormal_b = abnormal_b[abnormal_b["service"].isin(services)]
    base = (
        normal_b.groupby(["service", "feature"])["value"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "base_mean", "std": "base_std"})
    )
    base["base_std"] = base["base_std"].fillna(0.0)
    abnormal_b["ts_bin"] = (abnormal_b["ts"] // freq_s) * freq_s
    agg = (
        abnormal_b.groupby(["ts_bin", "service", "feature"])["value"]
        .mean()
        .reset_index()
        .rename(columns={"ts_bin": "ts"})
    )
    agg = agg.merge(base, on=["service", "feature"], how="left")
    agg["base_mean"] = agg["base_mean"].fillna(0.0)
    agg["base_std"] = agg["base_std"].fillna(0.0)
    eps = 1e-6
    agg["z"] = (agg["value"] - agg["base_mean"]) / (agg["base_std"] + eps)
    cols = []
    for s in services:
        for fk in FEATURE_KEYS:
            cols.append(f"{s}::{fk}")
    out = pd.DataFrame({"ts": ts_index})
    out.set_index("ts", inplace=True)
    pivot = agg.pivot_table(
        index="ts", columns=["service", "feature"], values="z", aggfunc="mean"
    )
    for s in services:
        for fk in FEATURE_KEYS:
            if (s, fk) in pivot.columns:
                out[f"{s}::{fk}"] = pivot[(s, fk)]
            else:
                out[f"{s}::{fk}"] = 0.0
    out = out.fillna(0.0).reset_index()
    return out, services


def _compute_service_delta_metrics(
    normal_long: pd.DataFrame,
    abnormal_long: pd.DataFrame,
    normal_range: Tuple[int, int],
    abnormal_range: Tuple[int, int],
) -> pd.DataFrame:
    n0, n1 = normal_range
    a0, a1 = abnormal_range

    def add_feature_bucket(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["feature"] = df["metric"].map(_map_metric_to_feature)
        df = df.dropna(subset=["feature"])
        return df

    normal_b = add_feature_bucket(normal_long)
    abnormal_b = add_feature_bucket(abnormal_long)
    normal_b = normal_b[(normal_b["ts"] >= n0) & (normal_b["ts"] <= n1)]
    abnormal_b = abnormal_b[(abnormal_b["ts"] >= a0) & (abnormal_b["ts"] <= a1)]
    if normal_b.empty and abnormal_b.empty:
        return pd.DataFrame(
            columns=[
                "service",
                "feature",
                "delta",
                "diff",
                "z",
                "robust_z",
                "log_ratio",
            ]
        )

    def _mad(x: pd.Series) -> float:
        med = float(x.median()) if len(x) else 0.0
        return float((x - med).abs().median()) if len(x) else 0.0

    n_agg = (
        normal_b.groupby(["service", "feature"])["value"]
        .agg(
            normal_mean="mean",
            normal_std="std",
            normal_median="median",
            normal_mad=_mad,
        )
        .reset_index()
    )
    a_agg = (
        abnormal_b.groupby(["service", "feature"])["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "abnormal_mean"})
    )
    df = a_agg.merge(n_agg, on=["service", "feature"], how="left")
    df["normal_mean"] = df["normal_mean"].fillna(0.0)
    df["normal_std"] = df["normal_std"].fillna(0.0)
    df["normal_mad"] = df["normal_mad"].fillna(0.0)
    eps = 1e-6
    df["diff"] = df["abnormal_mean"] - df["normal_mean"]
    df["z"] = df["diff"] / (df["normal_std"].abs() + eps)
    df["robust_z"] = df["diff"] / (df["normal_mad"].abs() + eps)
    ratio = (df["abnormal_mean"].abs() + eps) / (df["normal_mean"].abs() + eps)
    df["log_ratio"] = np.log(ratio.to_numpy(dtype=np.float64)).astype(np.float32)
    df["delta"] = df["robust_z"].clip(lower=-10.0, upper=10.0)
    return df[["service", "feature", "delta", "diff", "z", "robust_z", "log_ratio"]]


def _compute_trace_service_scores(conclusion_parquet: Path) -> pd.DataFrame:
    try:
        df = pd.read_parquet(conclusion_parquet)
    except Exception:
        return pd.DataFrame(columns=["service", "latency_ratio", "succ_drop"])
    if df is None or df.empty:
        return pd.DataFrame(columns=["service", "latency_ratio", "succ_drop"])
    if "SpanName" not in df.columns:
        return pd.DataFrame(columns=["service", "latency_ratio", "succ_drop"])
    df = df.copy()
    df["service"] = df["SpanName"].map(_parse_service_from_span)
    df = df.dropna(subset=["service"])
    if df.empty:
        return pd.DataFrame(columns=["service", "latency_ratio", "succ_drop"])
    eps = 1e-6
    if "AbnormalAvgDuration" in df.columns and "NormalAvgDuration" in df.columns:
        df["latency_ratio"] = df["AbnormalAvgDuration"] / (
            df["NormalAvgDuration"] + eps
        )
    else:
        df["latency_ratio"] = 1.0
    if "AbnormalSuccRate" in df.columns and "NormalSuccRate" in df.columns:
        df["succ_drop"] = df["NormalSuccRate"] - df["AbnormalSuccRate"]
    else:
        df["succ_drop"] = 0.0
    agg = df.groupby("service")[["latency_ratio", "succ_drop"]].max().reset_index()
    return agg


def _build_edges_from_traces(
    trace_parquet: Path, services: List[str]
) -> List[Tuple[str, str]]:
    try:
        df = pd.read_parquet(trace_parquet)
    except Exception:
        return []
    if df.empty:
        return []
    caller_col = _detect_col(
        df.columns,
        ("caller_service", "src_service", "source_service", "parent_service", "caller"),
    )
    callee_col = _detect_col(
        df.columns,
        ("callee_service", "dst_service", "target_service", "child_service", "callee"),
    )
    if not caller_col or not callee_col:
        span_col = _detect_col(df.columns, ("span", "span_name", "operation", "name"))
        if not span_col:
            return []
        return []
    sub = df[[caller_col, callee_col]].dropna()
    sub = sub.rename(columns={caller_col: "u", callee_col: "v"})
    sub["u"] = sub["u"].astype("string")
    sub["v"] = sub["v"].astype("string")
    sub = sub[sub["u"].isin(services) & sub["v"].isin(services)]
    edges = list({(r["u"], r["v"]) for _, r in sub.iterrows() if r["u"] != r["v"]})
    return edges


def _extract_services_and_edges_from_traces(
    trace_parquet: Path,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    try:
        df = pd.read_parquet(trace_parquet)
    except Exception:
        return [], []
    if df is None or df.empty:
        return [], []
    caller_col = _detect_col(
        df.columns,
        (
            "caller_service",
            "src_service",
            "source_service",
            "parent_service",
            "caller",
            "upstream_service",
        ),
    )
    callee_col = _detect_col(
        df.columns,
        (
            "callee_service",
            "dst_service",
            "target_service",
            "child_service",
            "callee",
            "downstream_service",
        ),
    )
    if caller_col and callee_col:
        sub = df[[caller_col, callee_col]].dropna()
        sub = sub.rename(columns={caller_col: "u", callee_col: "v"})
        sub["u"] = sub["u"].astype("string")
        sub["v"] = sub["v"].astype("string")
        sub = sub[(sub["u"].str.len() > 0) & (sub["v"].str.len() > 0)]
        services = sorted(set(sub["u"].tolist()) | set(sub["v"].tolist()))
        edges = list({(r["u"], r["v"]) for _, r in sub.iterrows() if r["u"] != r["v"]})
        return services, edges
    trace_id_col = _detect_col(df.columns, ("trace_id", "traceId"))
    span_id_col = _detect_col(df.columns, ("span_id", "spanId", "id"))
    parent_span_id_col = _detect_col(
        df.columns, ("parent_span_id", "parentSpanId", "parent_id")
    )
    service_col = _detect_col(
        df.columns, ("service_name", "serviceName", "attr.k8s.service.name")
    )
    if not service_col:
        service_col = _detect_col(df.columns, ("attr.k8s.service.name",))
    if trace_id_col and span_id_col and parent_span_id_col and service_col:
        sub = (
            df[[trace_id_col, span_id_col, parent_span_id_col, service_col]]
            .dropna(subset=[trace_id_col, span_id_col, service_col])
            .copy()
        )
        sub = sub.rename(
            columns={
                trace_id_col: "trace_id",
                span_id_col: "span_id",
                parent_span_id_col: "parent_span_id",
                service_col: "service",
            }
        )
        sub["trace_id"] = sub["trace_id"].astype("string")
        sub["span_id"] = sub["span_id"].astype("string")
        sub["parent_span_id"] = sub["parent_span_id"].astype("string")
        sub["service"] = sub["service"].astype("string")
        sub = sub[sub["service"].str.len() > 0]
        services = sorted(set(sub["service"].dropna().tolist()))
        parents = sub[["trace_id", "span_id", "service"]].rename(
            columns={"span_id": "parent_span_id", "service": "parent_service"}
        )
        joined = sub.merge(parents, on=["trace_id", "parent_span_id"], how="left")
        joined = joined.dropna(subset=["parent_service"])
        joined = joined.rename(columns={"service": "child_service"})
        edges = list(
            {
                (r["parent_service"], r["child_service"])
                for _, r in joined.iterrows()
                if r["parent_service"] != r["child_service"]
            }
        )
        return services, edges
    return [], []


def preprocess_one_scenario(
    scenario_dir: Path,
    out_dir: Path,
    freq_s: int,
    enable_mapping_debug: bool = False,
) -> Optional[Dict[str, Any]]:
    env = _read_json(scenario_dir / "env.json")
    inj = _read_json(scenario_dir / "injection.json")
    normal_start = _safe_int(env.get("NORMAL_START"))
    normal_end = _safe_int(env.get("NORMAL_END"))
    abnormal_start = _safe_int(env.get("ABNORMAL_START"))
    abnormal_end = _safe_int(env.get("ABNORMAL_END"))
    if None in (normal_start, normal_end, abnormal_start, abnormal_end):
        raise ValueError(f"env.json 缺少时间窗字段: {scenario_dir}")
    gt_service = None
    gt_services: List[str] = []
    try:
        gt_service = (inj.get("ground_truth", {}) or {}).get("service", [None])[0]
    except Exception:
        gt_service = None
    try:
        raw_gt_services = (inj.get("ground_truth", {}) or {}).get("service", None)
        if isinstance(raw_gt_services, list):
            for x in raw_gt_services:
                cx = canonicalize_service_name(x)
                if cx and cx not in gt_services:
                    gt_services.append(cx)
    except Exception:
        pass
    c1 = canonicalize_service_name(gt_service)
    if c1 and c1 not in gt_services:
        gt_services.insert(0, c1)
    gt_service = gt_services[0] if gt_services else None
    normal_metrics_pq = scenario_dir / "normal_metrics.parquet"
    abnormal_metrics_pq = scenario_dir / "abnormal_metrics.parquet"
    if not normal_metrics_pq.exists() or not abnormal_metrics_pq.exists():
        raise FileNotFoundError(f"缺少 metrics parquet: {scenario_dir}")
    normal_long = _load_metrics_long(normal_metrics_pq)
    abnormal_long = _load_metrics_long(abnormal_metrics_pq)
    mapping_debug = (
        _build_mapping_debug(scenario_dir.name, normal_long, abnormal_long)
        if enable_mapping_debug
        else None
    )
    services_from_traces: List[str] = []
    edges_from_traces: List[Tuple[str, str]] = []
    abnormal_traces_pq = scenario_dir / "abnormal_traces.parquet"
    if abnormal_traces_pq.exists():
        services_from_traces, edges_from_traces = (
            _extract_services_and_edges_from_traces(abnormal_traces_pq)
        )
    if not services_from_traces:
        normal_traces_pq = scenario_dir / "normal_traces.parquet"
        if normal_traces_pq.exists():
            services_from_traces, edges_from_traces = (
                _extract_services_and_edges_from_traces(normal_traces_pq)
            )
    service_ts, services = _compute_service_timeseries(
        normal_long=normal_long,
        abnormal_long=abnormal_long,
        normal_range=(normal_start, normal_end),
        abnormal_range=(abnormal_start, abnormal_end),
        freq_s=freq_s,
        gt_service=gt_service,
    )
    if gt_services:
        for gts in gt_services:
            if gts and gts not in services:
                services = [gts] + services
        services = services[:200]
        if not service_ts.empty:
            ts_col = service_ts["ts"]
            cols: Dict[str, pd.Series] = {"ts": ts_col}
            for s in services:
                for fk in FEATURE_KEYS:
                    col = f"{s}::{fk}"
                    cols[col] = service_ts[col] if col in service_ts.columns else 0.0
            service_ts = pd.DataFrame(cols)
    if services_from_traces:
        merged: List[str] = []
        if gt_service:
            merged.append(gt_service)
        for s in services_from_traces:
            if s and s not in merged:
                merged.append(s)
        for s in services:
            if s and s not in merged:
                merged.append(s)
        services = merged[:200]
        if not service_ts.empty:
            ts_col = service_ts["ts"]
            cols2: Dict[str, pd.Series] = {"ts": ts_col}
            for s in services:
                for fk in FEATURE_KEYS:
                    col = f"{s}::{fk}"
                    cols2[col] = service_ts[col] if col in service_ts.columns else 0.0
            service_ts = pd.DataFrame(cols2)
    delta_df = _compute_service_delta_metrics(
        normal_long=normal_long,
        abnormal_long=abnormal_long,
        normal_range=(normal_start, normal_end),
        abnormal_range=(abnormal_start, abnormal_end),
    )
    if not delta_df.empty:
        delta_pivot = delta_df.pivot_table(
            index="service", columns="feature", values="delta", aggfunc="mean"
        ).reset_index()
    else:
        delta_pivot = pd.DataFrame(columns=["service"] + list(FEATURE_KEYS))
    rows = []
    for s in services:
        row = {"service": s}
        sub = delta_pivot[delta_pivot["service"] == s]
        for fk in FEATURE_KEYS:
            if not sub.empty and fk in sub.columns:
                row[fk] = float(sub[fk].iloc[0])
            else:
                row[fk] = 0.0
        rows.append(row)
    delta_out = pd.DataFrame(rows, columns=["service"] + list(FEATURE_KEYS))
    failure_type_str = infer_rcabench_failure_type(scenario_dir.name, inj)
    edges: List[Tuple[str, str]] = []
    if edges_from_traces:
        edges = [
            (u, v) for (u, v) in edges_from_traces if u in services and v in services
        ]
    else:
        if abnormal_traces_pq.exists():
            edges = _build_edges_from_traces(abnormal_traces_pq, services)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "scenario": scenario_dir.name,
        "normal_range": [normal_start, normal_end],
        "abnormal_range": [abnormal_start, abnormal_end],
        "freq_s": int(freq_s),
        "ground_truth": inj.get("ground_truth", {}),
        "fault_type": inj.get("fault_type"),
        "failure_type": failure_type_str,
        "start_time": inj.get("start_time"),
        "end_time": inj.get("end_time"),
        "services": services,
    }
    try:
        if "ground_truth" not in meta or meta["ground_truth"] is None:
            meta["ground_truth"] = {}
        if isinstance(meta["ground_truth"], dict):
            meta["ground_truth"]["service"] = (
                gt_services
                if gt_services
                else (meta["ground_truth"].get("service") or [])
            )
    except Exception:
        pass
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    nodes_df = pd.DataFrame(
        {
            "node_id": list(range(len(services))),
            "node_type": ["pod"] * len(services),
            "node_name": services,
        }
    )
    nodes_df.to_csv(out_dir / "nodes.csv", index=False)
    edge_rows = []
    for u, v in edges:
        if u in services and v in services:
            edge_rows.append(
                {
                    "edge_type": "pod2pod",
                    "src": services.index(u),
                    "dst": services.index(v),
                }
            )
    edges_df = pd.DataFrame(edge_rows, columns=["edge_type", "src", "dst"])
    edges_df.to_csv(out_dir / "edges.csv", index=False)
    service_ts.to_csv(out_dir / "service_timeseries.csv", index=False)
    delta_out.to_csv(out_dir / "service_delta_metrics.csv", index=False)
    conclusion_pq = scenario_dir / "conclusion.parquet"
    if conclusion_pq.exists():
        trace_scores = _compute_trace_service_scores(conclusion_pq)
        rows_ts = []
        for s in services:
            row = {"service": s, "latency_ratio": 0.0, "succ_drop": 0.0}
            sub = trace_scores[trace_scores["service"] == s]
            if not sub.empty:
                row["latency_ratio"] = float(sub["latency_ratio"].iloc[0])
                row["succ_drop"] = float(sub["succ_drop"].iloc[0])
            rows_ts.append(row)
        trace_out = pd.DataFrame(
            rows_ts, columns=["service", "latency_ratio", "succ_drop"]
        )
        trace_out.to_csv(out_dir / "trace_service_scores.csv", index=False)
    return mapping_debug


def _aggregate_mapping_debug(all_debug: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not all_debug:
        return {"summary": {"scenarios": 0}, "scenarios": []}

    def _sum_bucket(path: str) -> Dict[str, int]:
        out = {k: 0 for k in FEATURE_KEYS}
        for d in all_debug:
            cur = d
            for p in path.split("."):
                cur = cur.get(p, {})
            for k in FEATURE_KEYS:
                out[k] += int(cur.get(k, 0))
        return out

    rows_ref = _sum_bucket("rows.bucket_counts_reference")
    rows_cur = _sum_bucket("rows.bucket_counts_current")
    rows_delta = {k: int(rows_cur[k] - rows_ref[k]) for k in FEATURE_KEYS}
    uniq_ref = _sum_bucket("unique_metrics.bucket_counts_reference")
    uniq_cur = _sum_bucket("unique_metrics.bucket_counts_current")
    uniq_delta = {k: int(uniq_cur[k] - uniq_ref[k]) for k in FEATURE_KEYS}
    unmapped_rows_ref = int(
        sum(d["rows"]["unmapped_hits"]["reference"] for d in all_debug)
    )
    unmapped_rows_cur = int(
        sum(d["rows"]["unmapped_hits"]["current"] for d in all_debug)
    )
    unmapped_uniq_ref = int(
        sum(d["unique_metrics"]["unmapped_hits"]["reference"] for d in all_debug)
    )
    unmapped_uniq_cur = int(
        sum(d["unique_metrics"]["unmapped_hits"]["current"] for d in all_debug)
    )
    return {
        "summary": {
            "scenarios": len(all_debug),
            "rows_bucket_counts_reference": rows_ref,
            "rows_bucket_counts_current": rows_cur,
            "rows_bucket_delta_current_minus_reference": rows_delta,
            "rows_unmapped_reference": unmapped_rows_ref,
            "rows_unmapped_current": unmapped_rows_cur,
            "rows_unmapped_delta_current_minus_reference": int(
                unmapped_rows_cur - unmapped_rows_ref
            ),
            "unique_bucket_counts_reference": uniq_ref,
            "unique_bucket_counts_current": uniq_cur,
            "unique_bucket_delta_current_minus_reference": uniq_delta,
            "unique_unmapped_reference": unmapped_uniq_ref,
            "unique_unmapped_current": unmapped_uniq_cur,
            "unique_unmapped_delta_current_minus_reference": int(
                unmapped_uniq_cur - unmapped_uniq_ref
            ),
        },
        "scenarios": all_debug,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess RCAbench for RippleRCA")
    p.add_argument(
        "--rcabench_root",
        type=str,
        required=True,
        help="RCAbench 根目录（包含 scenarios/）",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="预处理输出根目录（默认：<rcabench_root>/rcabench_preprocessed）",
    )
    p.add_argument("--freq_s", type=int, default=10, help="重采样频率（秒）")
    p.add_argument(
        "--max_scenarios",
        type=int,
        default=10_000,
        help="最多处理多少个 scenario（调试用）",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip scenarios that already have complete preprocessed core files.",
    )
    p.add_argument(
        "--mapping_debug_path",
        type=str,
        default=None,
        help="输出映射调试 json 路径（可选，开启后会统计场景级+全局级映射差分）",
    )
    return p.parse_args()


def _preprocessed_scenario_complete(out_dir: Path) -> bool:
    required = (
        "meta.json",
        "nodes.csv",
        "edges.csv",
        "service_timeseries.csv",
        "service_delta_metrics.csv",
    )
    return all((out_dir / name).exists() for name in required)


def main() -> None:
    _require_parquet_engine()
    args = parse_args()
    rcabench_root = Path(args.rcabench_root)
    scenarios_root = rcabench_root / "scenarios"
    source_layout = "nested_scenarios_dir"
    if not scenarios_root.exists():
        direct_scenarios = (
            [
                p
                for p in rcabench_root.iterdir()
                if p.is_dir()
                and (p / "injection.json").exists()
                and (p / "env.json").exists()
            ]
            if rcabench_root.exists()
            else []
        )
        if direct_scenarios:
            scenarios_root = rcabench_root
            source_layout = "direct_scenario_dirs"
    if not scenarios_root.exists():
        raise FileNotFoundError(f"未找到 scenarios/: {scenarios_root}")
    output_root = (
        Path(args.output_root)
        if args.output_root
        else (rcabench_root / "rcabench_preprocessed")
    )
    out_scenarios = output_root / "scenarios"
    out_scenarios.mkdir(parents=True, exist_ok=True)
    scenario_dirs = sorted(
        [
            p
            for p in scenarios_root.iterdir()
            if p.is_dir()
            and (p / "injection.json").exists()
            and (p / "env.json").exists()
        ]
    )
    scenario_dirs = scenario_dirs[: int(args.max_scenarios)]
    ok, fail, skipped = 0, 0, 0
    all_mapping_debug: List[Dict[str, Any]] = []
    for sdir in scenario_dirs:
        scenario_out_dir = out_scenarios / sdir.name
        if bool(args.skip_existing) and _preprocessed_scenario_complete(
            scenario_out_dir
        ):
            skipped += 1
            ok += 1
            continue
        try:
            md = preprocess_one_scenario(
                sdir,
                scenario_out_dir,
                freq_s=int(args.freq_s),
                enable_mapping_debug=bool(args.mapping_debug_path),
            )
            if md is not None:
                all_mapping_debug.append(md)
            ok += 1
        except Exception as e:
            fail += 1
            scenario_out_dir.mkdir(parents=True, exist_ok=True)
            (scenario_out_dir / "FAILED.txt").write_text(str(e), encoding="utf-8")
    summary = {
        "total": len(scenario_dirs),
        "ok": ok,
        "fail": fail,
        "skipped_existing": skipped,
        "source_layout": source_layout,
        "source_root": str(rcabench_root),
        "scenarios_root": str(scenarios_root),
        "output_root": str(output_root),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.mapping_debug_path:
        md_out = Path(args.mapping_debug_path)
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(
            json.dumps(
                _aggregate_mapping_debug(all_mapping_debug),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
