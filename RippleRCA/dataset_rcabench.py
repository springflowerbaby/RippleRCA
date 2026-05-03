from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch as th
import dgl
from dataset import myDGLDataset
from log import Logger

logger = Logger(__name__)
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
RAW_STAGE1_EXTRA_DIM = 16


def _canonicalize_service_name(name: Optional[str]) -> Optional[str]:
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
    if not s:
        return None
    parts = s.split("-")
    if len(parts) >= 3 and parts[-1].isalnum() and 4 <= len(parts[-1]) <= 10:
        if any(ch.isdigit() for ch in parts[-2]) or len(parts[-2]) >= 6:
            s = "-".join(parts[:-2])
    return s or None


class DatasetRCAbench(myDGLDataset):
    def __init__(
        self,
        preprocessed_root: str,
        scenarios: Optional[List[str]] = None,
        max_samples: int = 1_000_000,
        failure_types: Optional[List[str]] = None,
        add_self_loop: bool = True,
        **kwargs,
    ) -> None:
        self.preprocessed_root = Path(preprocessed_root)
        self.scenarios: Optional[List[str]] = scenarios
        self.raw_scenarios_root = self.preprocessed_root.parent / "scenarios"
        super().__init__(
            dataset_name="rcabench",
            data_dir=str(self.preprocessed_root),
            dates=scenarios or ["_scenarios_"],
            node_feature_selector={"pod": ["in_degree", "out_degree", "metric_z8"]},
            edge_reverse=False,
            add_self_loop=add_self_loop,
            max_samples=max_samples,
            failure_types=failure_types or [""],
            special_failure_types=[],
            failure_duration=5,
            is_mask=False,
            process_miss="interpolate",
            process_extreme=False,
            k_sigma=3,
            use_split_info=False,
        )

    def pod_to_service(self, pod_name: str) -> str:
        return str(pod_name)

    def get_stacked_nfeat(self):
        if len(self.graphs) == 0:
            return {"pod": th.tensor([])}
        feats = []
        for g in self.graphs:
            feats.append(g.nodes["pod"].data["feat"])
        return {"pod": th.stack(feats, dim=0)}

    def get_nan_nodes(self):
        if hasattr(self, "nan_nodes"):
            return self.nan_nodes
        self.nan_nodes = {}
        if len(self.graphs) == 0:
            self.nan_nodes["pod"] = th.tensor([])
            return self.nan_nodes
        nans = []
        for g in self.graphs:
            data = g.nodes["pod"].data["feat"]
            nan = th.where(th.isnan(data).all(dim=1), th.tensor(True), th.tensor(False))
            g.nodes["pod"].data["nan"] = nan.reshape(-1, 1)
            nans.append(nan)
        self.nan_nodes["pod"] = th.stack(nans, dim=0)
        return self.nan_nodes

    def get_nan_edges(self):
        if hasattr(self, "nan_edges"):
            return self.nan_edges
        self.nan_edges = {}
        etype = ("pod", "calls", "pod")
        self.nan_edges[etype] = th.tensor([])
        return self.nan_edges

    def _list_scenarios(self) -> List[str]:
        sroot = self.preprocessed_root / "scenarios"
        if not sroot.exists():
            raise FileNotFoundError(f"Preprocessing directory not found scenarios/: {sroot}")
        all_s = sorted([p.name for p in sroot.iterdir() if p.is_dir()])
        if self.scenarios:
            wanted = []
            for s in self.scenarios:
                if (sroot / s).exists():
                    wanted.append(s)
                else:
                    logger.warning(f"scenario Does not exist (skip): {s}")
            return wanted
        return all_s

    def _read_meta(self, scenario: str) -> Dict:
        p = self.preprocessed_root / "scenarios" / scenario / "meta.json"
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _read_nodes_edges(
        self, scenario: str
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        sdir = self.preprocessed_root / "scenarios" / scenario
        nodes = pd.read_csv(sdir / "nodes.csv")
        if "node_name" not in nodes.columns:
            raise ValueError(f"nodes.csv less node_name: {sdir}")
        services = nodes["node_name"].astype(str).tolist()
        edges_path = sdir / "edges.csv"
        edges: List[Tuple[int, int]] = []
        if edges_path.exists():
            edf = pd.read_csv(edges_path)
            if set(["src", "dst"]).issubset(edf.columns):
                for _, r in edf.iterrows():
                    try:
                        u = int(r["src"])
                        v = int(r["dst"])
                        if 0 <= u < len(services) and 0 <= v < len(services):
                            edges.append((u, v))
                    except Exception:
                        continue
        return services, edges

    def _read_edges_as_names(self, scenario: str) -> List[Tuple[str, str]]:
        services, edges_idx = self._read_nodes_edges(scenario)
        edges_name: List[Tuple[str, str]] = []
        for u, v in edges_idx:
            try:
                su = services[u]
                sv = services[v]
                if su and sv and su != sv:
                    edges_name.append((su, sv))
            except Exception:
                continue
        return edges_name

    def _read_service_timeseries(self, scenario: str, services: List[str]) -> th.Tensor:
        sdir = self.preprocessed_root / "scenarios" / scenario
        df = pd.read_csv(sdir / "service_timeseries.csv")
        if df.empty or "ts" not in df.columns:
            return th.zeros((1, len(services), len(FEATURE_KEYS))).float()
        T = len(df)
        N = len(services)
        F = len(FEATURE_KEYS)
        arr = np.zeros((T, N, F), dtype=np.float32)
        for i, s in enumerate(services):
            for j, fk in enumerate(FEATURE_KEYS):
                col = f"{s}::{fk}"
                if col in df.columns:
                    v = (
                        pd.to_numeric(df[col], errors="coerce")
                        .fillna(0.0)
                        .to_numpy(dtype=np.float32)
                    )
                    if v.shape[0] == T:
                        arr[:, i, j] = v
        return th.from_numpy(arr).float()

    def _read_service_delta(self, scenario: str, services: List[str]) -> th.Tensor:
        sdir = self.preprocessed_root / "scenarios" / scenario
        path = sdir / "service_delta_metrics.csv"
        if not path.exists():
            return th.zeros((len(services), len(FEATURE_KEYS))).float()
        df = pd.read_csv(path)
        if df.empty or "service" not in df.columns:
            return th.zeros((len(services), len(FEATURE_KEYS))).float()
        df["service"] = df["service"].astype(str)
        arr = np.zeros((len(services), len(FEATURE_KEYS)), dtype=np.float32)
        for i, s in enumerate(services):
            sub = df[df["service"] == s]
            if sub.empty:
                continue
            for j, fk in enumerate(FEATURE_KEYS):
                if fk in sub.columns:
                    try:
                        arr[i, j] = float(sub[fk].iloc[0])
                    except Exception:
                        arr[i, j] = 0.0
        return th.from_numpy(arr).float()

    def _read_trace_scores(self, scenario: str, services: List[str]) -> th.Tensor:
        sdir = self.preprocessed_root / "scenarios" / scenario
        path = sdir / "trace_service_scores.csv"
        if not path.exists():
            return th.zeros((len(services), 2)).float()
        df = pd.read_csv(path)
        if df.empty or "service" not in df.columns:
            return th.zeros((len(services), 2)).float()
        df["service"] = df["service"].astype(str)
        arr = np.zeros((len(services), 2), dtype=np.float32)
        for i, s in enumerate(services):
            sub = df[df["service"] == s]
            if sub.empty:
                continue
            try:
                arr[i, 0] = float(sub.get("latency_ratio", 0.0).iloc[0])
            except Exception:
                arr[i, 0] = 0.0
            try:
                arr[i, 1] = float(sub.get("succ_drop", 0.0).iloc[0])
            except Exception:
                arr[i, 1] = 0.0
        return th.from_numpy(arr).float()

    def _resolve_metric_service_name(self, row: pd.Series) -> Optional[str]:
        for key in (
            "service_name",
            "attr.k8s.container.name",
            "attr.k8s.service.name",
            "attr.destination_workload",
            "attr.source_workload",
        ):
            if key in row and pd.notna(row[key]):
                sx = _canonicalize_service_name(row[key])
                if sx:
                    return sx
        if "attr.k8s.pod.name" in row and pd.notna(row["attr.k8s.pod.name"]):
            return _canonicalize_service_name(row["attr.k8s.pod.name"])
        return None

    def _load_raw_metrics_frame(self, scenario: str, abnormal: bool) -> pd.DataFrame:
        sdir = self.raw_scenarios_root / scenario
        fname = "abnormal_metrics.parquet" if abnormal else "normal_metrics.parquet"
        path = sdir / fname
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()
        if df.empty or "metric" not in df.columns or "value" not in df.columns:
            return pd.DataFrame()
        df = df.copy()
        df["metric"] = df["metric"].astype(str)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["metric", "value"])
        if df.empty:
            return pd.DataFrame()
        df["service"] = df.apply(self._resolve_metric_service_name, axis=1)
        df = df.dropna(subset=["service"])
        return df

    def _load_raw_traces_frame(self, scenario: str, abnormal: bool) -> pd.DataFrame:
        sdir = self.raw_scenarios_root / scenario
        fname = "abnormal_traces.parquet" if abnormal else "normal_traces.parquet"
        path = sdir / fname
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()
        if df.empty:
            return pd.DataFrame()
        service_col = None
        for key in ("service_name", "attr.k8s.service.name"):
            if key in df.columns:
                service_col = key
                break
        if service_col is None or "duration" not in df.columns:
            return pd.DataFrame()
        out = pd.DataFrame()
        out["service"] = df[service_col].map(_canonicalize_service_name)
        out["duration"] = pd.to_numeric(df["duration"], errors="coerce")
        status_series = None
        for key in ("attr.http.response.status_code", "attr.status_code"):
            if key in df.columns:
                status_series = df[key]
                break
        if status_series is None:
            out["status_bad"] = 0.0
        else:
            status_num = pd.to_numeric(status_series, errors="coerce")
            bad = ((status_num >= 500) | (status_num == 0)).astype(float)
            text_bad = (
                status_series.astype(str)
                .str.lower()
                .isin(["error", "status_code_error"])
                .astype(float)
            )
            out["status_bad"] = np.maximum(
                bad.to_numpy(dtype=np.float32), text_bad.to_numpy(dtype=np.float32)
            )
        out = out.dropna(subset=["service", "duration"])
        return out

    def _robust_metric_shift(
        self,
        normal_df: pd.DataFrame,
        abnormal_df: pd.DataFrame,
        services: List[str],
        metric_name: str,
        higher_is_worse: bool = True,
        absolute_diff: bool = False,
    ) -> th.Tensor:
        out = np.zeros((len(services),), dtype=np.float32)
        nd = (
            normal_df[normal_df["metric"] == metric_name]
            if not normal_df.empty
            else pd.DataFrame()
        )
        ad = (
            abnormal_df[abnormal_df["metric"] == metric_name]
            if not abnormal_df.empty
            else pd.DataFrame()
        )
        eps = 1e-6
        for i, service in enumerate(services):
            nvals = (
                nd.loc[nd["service"] == service, "value"].to_numpy(dtype=np.float32)
                if not nd.empty
                else np.array([], dtype=np.float32)
            )
            avals = (
                ad.loc[ad["service"] == service, "value"].to_numpy(dtype=np.float32)
                if not ad.empty
                else np.array([], dtype=np.float32)
            )
            if avals.size == 0 and nvals.size == 0:
                continue
            abnormal_mean = float(avals.mean()) if avals.size else 0.0
            normal_mean = float(nvals.mean()) if nvals.size else 0.0
            normal_median = float(np.median(nvals)) if nvals.size else normal_mean
            normal_mad = (
                float(np.median(np.abs(nvals - normal_median))) if nvals.size else 0.0
            )
            diff = abnormal_mean - normal_mean
            if not higher_is_worse:
                diff = -diff
            if absolute_diff:
                diff = abs(abnormal_mean - normal_mean)
            out[i] = float(np.clip(diff / (normal_mad + eps), -10.0, 10.0))
        return th.from_numpy(out).float()

    def _trace_proxy_shift(
        self,
        normal_df: pd.DataFrame,
        abnormal_df: pd.DataFrame,
        services: List[str],
        value_col: str,
    ) -> th.Tensor:
        out = np.zeros((len(services),), dtype=np.float32)
        if value_col not in {"duration", "status_bad"}:
            return th.from_numpy(out).float()
        eps = 1e-6
        for i, service in enumerate(services):
            nvals = (
                normal_df.loc[normal_df["service"] == service, value_col].to_numpy(
                    dtype=np.float32
                )
                if not normal_df.empty
                else np.array([], dtype=np.float32)
            )
            avals = (
                abnormal_df.loc[abnormal_df["service"] == service, value_col].to_numpy(
                    dtype=np.float32
                )
                if not abnormal_df.empty
                else np.array([], dtype=np.float32)
            )
            if avals.size == 0 and nvals.size == 0:
                continue
            abnormal_mean = float(avals.mean()) if avals.size else 0.0
            normal_mean = float(nvals.mean()) if nvals.size else 0.0
            normal_median = float(np.median(nvals)) if nvals.size else normal_mean
            normal_mad = (
                float(np.median(np.abs(nvals - normal_median))) if nvals.size else 0.0
            )
            diff = abnormal_mean - normal_mean
            out[i] = float(np.clip(diff / (normal_mad + eps), -10.0, 10.0))
        return th.from_numpy(out).float()

    def _read_raw_stage1_extra_features(
        self, scenario: str, services: List[str]
    ) -> th.Tensor:
        zeros = th.zeros((len(services), RAW_STAGE1_EXTRA_DIM)).float()
        if not self.raw_scenarios_root.exists():
            return zeros
        normal_df = self._load_raw_metrics_frame(scenario, abnormal=False)
        abnormal_df = self._load_raw_metrics_frame(scenario, abnormal=True)
        normal_trace_df = self._load_raw_traces_frame(scenario, abnormal=False)
        abnormal_trace_df = self._load_raw_traces_frame(scenario, abnormal=True)
        if (
            normal_df.empty
            and abnormal_df.empty
            and normal_trace_df.empty
            and abnormal_trace_df.empty
        ):
            return zeros
        stress_queue = self._robust_metric_shift(
            normal_df, abnormal_df, services, "queueSize"
        )
        stress_cpu_limit = self._robust_metric_shift(
            normal_df, abnormal_df, services, "k8s.pod.cpu_limit_utilization"
        )
        stress_mem_limit = self._robust_metric_shift(
            normal_df, abnormal_df, services, "k8s.pod.memory_limit_utilization"
        )
        stress_jvm_load = self._robust_metric_shift(
            normal_df, abnormal_df, services, "jvm.system.cpu.load_1m"
        )
        pod_ready_drop = self._robust_metric_shift(
            normal_df,
            abnormal_df,
            services,
            "k8s.container.ready",
            higher_is_worse=False,
        )
        pod_restart = self._robust_metric_shift(
            normal_df, abnormal_df, services, "k8s.container.restarts"
        )
        pod_phase = self._robust_metric_shift(
            normal_df, abnormal_df, services, "k8s.pod.phase", absolute_diff=True
        )
        pod_availability = self._robust_metric_shift(
            normal_df,
            abnormal_df,
            services,
            "k8s.statefulset.ready_pods",
            higher_is_worse=False,
        )
        network_p50 = self._robust_metric_shift(
            normal_df, abnormal_df, services, "hubble_http_request_duration_p50_seconds"
        )
        network_p90 = self._robust_metric_shift(
            normal_df, abnormal_df, services, "hubble_http_request_duration_p90_seconds"
        )
        network_trace_duration = self._trace_proxy_shift(
            normal_trace_df, abnormal_trace_df, services, "duration"
        )
        network_trace_error = self._trace_proxy_shift(
            normal_trace_df, abnormal_trace_df, services, "status_bad"
        )
        mysql_p95 = self._robust_metric_shift(
            normal_df, abnormal_df, services, "hubble_http_request_duration_p95_seconds"
        )
        mysql_p99 = self._robust_metric_shift(
            normal_df, abnormal_df, services, "hubble_http_request_duration_p99_seconds"
        )
        mysql_trace_duration = self._trace_proxy_shift(
            normal_trace_df, abnormal_trace_df, services, "duration"
        )
        mysql_trace_error = self._trace_proxy_shift(
            normal_trace_df, abnormal_trace_df, services, "status_bad"
        )
        extra = th.stack(
            [
                stress_queue,
                stress_cpu_limit,
                stress_mem_limit,
                stress_jvm_load,
                network_p50,
                network_p90,
                network_trace_duration,
                network_trace_error,
                pod_ready_drop,
                pod_restart,
                pod_phase,
                pod_availability,
                mysql_p95,
                mysql_p99,
                mysql_trace_duration,
                mysql_trace_error,
            ],
            dim=1,
        )
        return extra.float()

    def get_stage1_channel_aux(self) -> th.Tensor:
        if len(self.graphs) == 0:
            return th.zeros((0, 0, RAW_STAGE1_EXTRA_DIM)).float()
        num_nodes = int(self.num_nodes_dict.get("pod", 0))
        services = [
            str(self.name_mapping.get(("pod", i), "unknown")) for i in range(num_nodes)
        ]
        aux_list: List[th.Tensor] = []
        for label in self.labels:
            scenario = str(label.get("scenario", "") or "")
            aux = self._read_raw_stage1_extra_features(scenario, services)
            if aux.shape[0] != num_nodes:
                fixed = th.zeros((num_nodes, RAW_STAGE1_EXTRA_DIM)).float()
                rows = min(num_nodes, int(aux.shape[0])) if aux.ndim == 2 else 0
                if rows > 0:
                    fixed[:rows] = aux[:rows]
                aux = fixed
            aux_list.append(aux.float())
        return (
            th.stack(aux_list, dim=0)
            if aux_list
            else th.zeros((0, num_nodes, RAW_STAGE1_EXTRA_DIM)).float()
        )

    def process(self) -> None:
        self.graphs = []
        self.labels = []
        self.groundtruths = []
        self.ntypes = ["pod"]
        self.etypes = [("pod", "calls", "pod")]
        scenario_list = self._list_scenarios()
        if not scenario_list:
            raise ValueError("No scenario was found. Please check the preprocess output directory.")
        global_services: List[str] = []
        for scenario in scenario_list:
            try:
                sdir = self.preprocessed_root / "scenarios" / scenario
                nodes = pd.read_csv(sdir / "nodes.csv")
                if "node_name" not in nodes.columns:
                    continue
                for s in nodes["node_name"].astype(str).tolist():
                    if s and s not in global_services:
                        global_services.append(s)
            except Exception:
                continue
        if "unknown" in global_services:
            global_services = [s for s in global_services if s != "unknown"] + [
                "unknown"
            ]
        if not global_services:
            global_services = ["unknown"]
        if len(global_services) > 300:
            global_services = global_services[:300]
        global_index = {s: i for i, s in enumerate(global_services)}
        for scenario in scenario_list:
            meta = self._read_meta(scenario)
            scenario_services, _ = self._read_nodes_edges(scenario)
            edges_name = self._read_edges_as_names(scenario)
            src_list: List[int] = []
            dst_list: List[int] = []
            for su, sv in edges_name:
                if su in global_index and sv in global_index and su != sv:
                    src_list.append(global_index[su])
                    dst_list.append(global_index[sv])
            src = th.tensor(src_list, dtype=th.int64)
            dst = th.tensor(dst_list, dtype=th.int64)
            g = dgl.heterograph(
                {("pod", "calls", "pod"): (src, dst)},
                num_nodes_dict={"pod": len(global_services)},
            )
            if self.add_self_loop:
                g = dgl.add_self_loop(g, etype=("pod", "calls", "pod"))
            service_ts = self._read_service_timeseries(scenario, global_services)
            if (
                service_ts.dim() != 3
                or service_ts.shape[1] != len(global_services)
                or service_ts.shape[2] != len(FEATURE_KEYS)
            ):
                service_ts = th.zeros(
                    (1, len(global_services), len(FEATURE_KEYS))
                ).float()
            feat_last = service_ts[-1]
            feat_delta = self._read_service_delta(scenario, global_services)
            feat_trace = self._read_trace_scores(scenario, global_services)
            feat_metric = th.cat([feat_last, feat_delta, feat_trace], dim=-1)
            in_deg = g.in_degrees(etype=("pod", "calls", "pod")).float().unsqueeze(-1)
            out_deg = g.out_degrees(etype=("pod", "calls", "pod")).float().unsqueeze(-1)
            feat = th.cat([in_deg, out_deg, feat_metric], dim=-1)
            g.nodes["pod"].data["feat"] = feat
            gt_services: List[str] = []
            try:
                raw = (meta.get("ground_truth", {}) or {}).get("service", None)
                if isinstance(raw, list):
                    for x in raw:
                        sx = str(x).strip()
                        if sx and sx not in gt_services:
                            gt_services.append(sx)
                elif isinstance(raw, str):
                    sx = raw.strip()
                    if sx:
                        gt_services = [sx]
            except Exception:
                gt_services = []
            if not gt_services:
                gt_services = ["unknown"]
            self.name_mapping.update(
                {("pod", i): s for i, s in enumerate(global_services)}
            )
            self.num_nodes_dict = {"pod": len(global_services)}
            gt_ids: List[int] = []
            for s in gt_services:
                if s in global_index:
                    gt_ids.append(global_index[s])
            gt_set = set([("pod", i) for i in gt_ids]) if gt_ids else set()
            label = {
                "scenario": scenario,
                "timestamp": int((meta.get("abnormal_range") or [0, 0])[0]),
                "level": "service",
                "cmdb_id": gt_services[0],
                "failure_type": str(
                    meta.get("failure_type", meta.get("fault_type", ""))
                ),
            }
            self.graphs.append(g)
            self.labels.append(label)
            self.groundtruths.append(gt_set)
            if len(self.graphs) >= self.max_samples:
                break
        if len(self.graphs) == 0:
            raise ValueError("RCAbench: No graph samples were generated.")
        self.get_nan_nodes()
        self.get_nan_edges()
        assert len(self.graphs) == len(self.labels) == len(self.groundtruths)
