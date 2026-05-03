import pickle
import csv
import os
import json

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable=None, *args, **kwargs):
        return iterable


from copy import deepcopy
from collections import defaultdict
from typing import List, Dict, Callable, Optional, Union, Tuple, Set, Iterable
import pandas as pd
import numpy as np
import networkx as nx
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch import GATConv

try:
    from stage2_reranker_split_eval import (
        BASE_FEATURE_COLS as STAGE2_RERANKER_BASE_FEATURE_COLS,
        _assign_hybrid_scores as _stage2_assign_hybrid_scores,
        _prepare_feature_frame as _stage2_prepare_feature_frame,
    )
except Exception:
    STAGE2_RERANKER_BASE_FEATURE_COLS = [
        "stage2_score",
        "stage1_score_sample",
        "stage1_mean_score",
        "stage1_vote_count",
        "stage1_peak_score",
    ]
    _stage2_assign_hybrid_scores = None
    _stage2_prepare_feature_frame = None
try:
    from mask import load_mask_feat, replace_node_feats
    from utils import euclidean, mahalanobis, load_stats
    from log import Logger
    from pcmci_plus_runner import run_pcmci_plus_from_stacked

    logger = Logger(__name__)
except ImportError as e:
    print(f"Warning: Could not import local RippleRCA modules: {e}")

    class Logger:
        def __init__(self, name):
            self.name = name

        def info(self, msg):
            print(f"[INFO] {msg}")

        def warning(self, msg):
            print(f"[WARN] {msg}")

        def error(self, msg):
            print(f"[ERROR] {msg}")

        def debug(self, msg):
            pass

    logger = Logger(__name__)
try:
    from tigramite.data_processing import DataFrame
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI
except ImportError:
    DataFrame = None
    ParCorr = None
    PCMCI = None


def constrained_pcmci_plus(
    g: DGLGraph,
    stacked_nfeat: Dict[str, th.Tensor],
    labels: List[Dict],
    ntype: str = "api",
    feat_idx: int = 0,
    tau_max: int = 3,
    alpha: float = 0.05,
    nan_nodes: Optional[Dict[str, th.Tensor]] = None,
) -> Dict[Tuple[int, int], Dict[int, float]]:
    logger.info(f"Stage 1.1: PCMCIplus (tau_max={tau_max})...")
    constrained_edges = set()
    for etype in g.canonical_etypes:
        if etype[2] == ntype:
            src, dst = g.edges(etype=etype)
            for u, v in zip(src.tolist(), dst.tolist()):
                constrained_edges.add((u, v))
    logger.info(f"Number of graph constraint edges:{len(constrained_edges)}")
    try:
        ts, keep_mask = _select_nodes_for_pcmci(
            stacked_nfeat, nan_nodes, ntype, feat_idx, None
        )
        if DataFrame is None or ParCorr is None or PCMCI is None:
            raise ImportError(
                "Tigramite not found, please install it first:pip install tigramite==5.2.9.4"
            )
        dataframe = DataFrame(ts)
        parcorr = ParCorr(significance="analytic")
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
        results = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=alpha)
        val_matrix = results["val_matrix"]
        q_matrix = results.get("q_matrix", None)
        p_matrix = results.get("p_matrix", None)
        original_ids = [i for i, k in enumerate(keep_mask) if k]
        P_prior = defaultdict(lambda: defaultdict(float))
        n_vars = ts.shape[1]
        for to_idx in range(n_vars):
            for frm_idx in range(n_vars):
                to_id = original_ids[to_idx]
                frm_id = original_ids[frm_idx]
                if (frm_id, to_id) not in constrained_edges:
                    continue
                lag_scores = []
                for lag in range(0, tau_max + 1):
                    val = val_matrix[to_idx, frm_idx, lag]
                    sig = (
                        q_matrix[to_idx, frm_idx, lag]
                        if q_matrix is not None
                        else p_matrix[to_idx, frm_idx, lag]
                    )
                    if sig is not None and not np.isnan(sig) and sig <= alpha:
                        significance_score = abs(float(val)) * (1.0 - sig / alpha)
                        lag_scores.append((lag, significance_score))
                if lag_scores:
                    total_score = sum(score for _, score in lag_scores)
                    if total_score > 0:
                        for lag, score in lag_scores:
                            P_prior[(frm_id, to_id)][lag] = score / total_score
        logger.info(f"Stage 1.1: Completed, discovered the time delay distribution of {len(P_prior)} constraint edges.")
        return dict(P_prior)
    except Exception as e:
        logger.warning(f"PCMCIplus failed to run: {e}, returning an empty time delay distribution.")
        return {}


def _select_nodes_for_pcmci(
    stacked_nfeat, nan_nodes, ntype: str, feat_idx: int, max_nodes: Optional[int]
):
    try:
        from pcmci_plus_runner import _select_nodes

        return _select_nodes(stacked_nfeat, nan_nodes, ntype, feat_idx, max_nodes)
    except ImportError:
        data = stacked_nfeat[ntype]
        if max_nodes is not None:
            data = data[:, :max_nodes, :]
        ts = data[:, :, feat_idx].detach().cpu().numpy()
        keep = np.ones(ts.shape[1], dtype=bool)
        for j in range(ts.shape[1]):
            if np.isnan(ts[:, j]).all():
                keep[j] = False
        ts = ts[:, keep]
        col_mean = np.nanmean(ts, axis=0)
        inds = np.where(np.isnan(ts))
        ts[inds] = np.take(col_mean, inds[1])
        return ts, keep


def constrained_pcmci_plus_with_timeseries(
    service_timeseries: th.Tensor,
    service_list: List[str],
    G_trace_edges: List[Tuple[str, str]],
    tau_max: int = 3,
    alpha: float = 0.05,
    feat_idx: int = 0,
) -> Dict[Tuple[int, int], Dict[int, float]]:
    logger.info(
        f"Stage 1.1: Running Constrained PCMCIplus (Service-Level Time Series)(tau_max={tau_max})..."
    )
    service_to_idx = {svc: i for i, svc in enumerate(service_list)}
    constrained_edges = set()
    for u, v in G_trace_edges:
        if u in service_to_idx and v in service_to_idx:
            u_idx = service_to_idx[u]
            v_idx = service_to_idx[v]
            constrained_edges.add((u_idx, v_idx))
    logger.info(f"Number of Trace topology constraint edges: {len(constrained_edges)}")
    ts = service_timeseries[:, :, feat_idx].detach().cpu().numpy()
    keep = np.ones(ts.shape[1], dtype=bool)
    for j in range(ts.shape[1]):
        if np.isnan(ts[:, j]).all():
            keep[j] = False
    ts = ts[:, keep]
    original_ids = [i for i, k in enumerate(keep) if k]
    col_mean = np.nanmean(ts, axis=0)
    inds = np.where(np.isnan(ts))
    if len(inds[0]) > 0:
        ts[inds] = np.take(col_mean, inds[1])
    try:
        if DataFrame is None or ParCorr is None or PCMCI is None:
            raise ImportError(
                "Tigramite not found. Please install it first: pip install tigramite==5.2.9.4"
            )
        dataframe = DataFrame(ts)
        parcorr = ParCorr(significance="analytic")
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
        results = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=alpha)
        val_matrix = results["val_matrix"]
        q_matrix = results.get("q_matrix", None)
        p_matrix = results.get("p_matrix", None)
        P_prior = defaultdict(lambda: defaultdict(float))
        n_vars = ts.shape[1]
        for to_idx in range(n_vars):
            for frm_idx in range(n_vars):
                to_id = original_ids[to_idx]
                frm_id = original_ids[frm_idx]
                if (frm_id, to_id) not in constrained_edges:
                    continue
                lag_scores = []
                for lag in range(0, tau_max + 1):
                    val = val_matrix[to_idx, frm_idx, lag]
                    sig = (
                        q_matrix[to_idx, frm_idx, lag]
                        if q_matrix is not None
                        else p_matrix[to_idx, frm_idx, lag]
                    )
                    if sig is not None and not np.isnan(sig) and sig <= alpha:
                        significance_score = abs(float(val)) * (1.0 - sig / alpha)
                        lag_scores.append((lag, significance_score))
                if lag_scores:
                    total_score = sum(score for _, score in lag_scores)
                    if total_score > 0:
                        for lag, score in lag_scores:
                            P_prior[(frm_id, to_id)][lag] = score / total_score
        logger.info(f"Stage 1.1: 完成，发现 {len(P_prior)} 条约束边的时滞分布")
        return dict(P_prior)
    except Exception as e:
        logger.warning(f"PCMCIplus运行失败: {e}，返回空时滞分布")
        import traceback

        logger.debug(traceback.format_exc())
        return {}


def historical_window_encoding(
    stacked_nfeat: Dict[str, th.Tensor],
    data_stats: Dict[str, Dict[str, th.Tensor]],
    tau_max: int,
    i: int,
    device: str = "cpu",
) -> Dict[str, Dict[int, th.Tensor]]:
    H = {}
    for ntype in stacked_nfeat.keys():
        H[ntype] = {}
        feat_seq = stacked_nfeat[ntype]
        norm_feat = data_stats.get("mean", {}).get(ntype, None)
        start_idx = max(0, i - tau_max)
        end_idx = i + 1
        for lag in range(tau_max + 1):
            hist_idx = i - lag
            if hist_idx < 0 or hist_idx >= feat_seq.shape[0]:
                H[ntype][lag] = th.zeros_like(feat_seq[0]).to(device)
            else:
                X_t_k = feat_seq[hist_idx].to(device)
                if norm_feat is not None:
                    X_norm = norm_feat.to(device)
                    if X_norm.dim() == 1 and X_norm.shape[0] == X_t_k.shape[0]:
                        H_diff = (
                            X_t_k - X_norm.unsqueeze(1)
                            if X_norm.dim() == 1
                            else X_t_k - X_norm
                        )
                    else:
                        H_diff = X_t_k - X_norm
                else:
                    H_diff = X_t_k
                if th.isnan(H_diff).any():
                    H_diff = th.where(th.isnan(H_diff), th.zeros_like(H_diff), H_diff)
                H[ntype][lag] = H_diff
    return H


def _safe_minmax_norm(scores: Optional[th.Tensor]) -> Optional[th.Tensor]:
    if scores is None:
        return None
    if scores.numel() == 0:
        return scores
    scores = scores.float()
    s_min = scores.min()
    s_max = scores.max()
    if float((s_max - s_min).abs().item()) < 1e-12:
        return th.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min + 1e-8)


def _tensor_to_score_list(scores: Optional[th.Tensor]) -> List[float]:
    if scores is None:
        return []
    return [float(x) for x in scores.detach().cpu().tolist()]


def _stable_rms_norm(values: th.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    values = values.float()
    if th.isnan(values).any():
        values = th.where(th.isnan(values), th.zeros_like(values), values)
    rms = th.sqrt(th.mean(values.pow(2)) + 1e-8)
    score = float(rms.item())
    if not np.isfinite(score):
        return 0.0
    return max(score, 0.0)


def _combine_score_tensors(
    score_items: List[Tuple[str, Optional[th.Tensor], float]],
    device: str,
) -> Optional[th.Tensor]:
    valid_items: List[Tuple[str, th.Tensor, float]] = []
    for name, scores, weight in score_items:
        if scores is None or weight <= 0:
            continue
        norm_scores = _safe_minmax_norm(scores)
        if norm_scores is None:
            continue
        valid_items.append((name, norm_scores.to(device), float(weight)))
    if not valid_items:
        return None
    total_weight = sum(weight for _, _, weight in valid_items)
    if total_weight <= 0:
        total_weight = float(len(valid_items))
        valid_items = [(name, scores, 1.0) for name, scores, _ in valid_items]
    fused = th.zeros_like(
        valid_items[0][1], dtype=th.float32, device=valid_items[0][1].device
    )
    for _, scores, weight in valid_items:
        fused = fused + (weight / total_weight) * scores
    return fused


def _transform_stage2_score_values(
    values: List[float],
    transform: str = "log1p_clip",
    clip_percentile: float = 95.0,
    clip_min_candidates: int = 8,
) -> Tuple[List[float], Dict[str, float]]:
    clean_values: List[float] = []
    for value in values:
        try:
            score = float(value)
        except Exception:
            score = 0.0
        if not np.isfinite(score):
            score = 0.0
        clean_values.append(max(score, 0.0))
    if not clean_values:
        return [], {
            "transform": str(transform),
            "clip_percentile": float(clip_percentile),
            "clip_threshold": 0.0,
            "raw_max": 0.0,
            "raw_p95": 0.0,
            "transformed_max": 0.0,
            "candidate_count": 0.0,
        }
    raw_max = float(max(clean_values))
    raw_p95 = float(np.percentile(clean_values, 95.0))
    clip_threshold = raw_max
    normalized_transform = str(transform or "none").strip().lower()
    if normalized_transform in ("clip", "log1p_clip") and len(clean_values) >= max(
        int(clip_min_candidates), 1
    ):
        bounded_percentile = min(max(float(clip_percentile), 0.0), 100.0)
        clip_threshold = float(np.percentile(clean_values, bounded_percentile))
    transformed: List[float] = []
    for score in clean_values:
        adjusted = score
        if normalized_transform in ("clip", "log1p_clip"):
            adjusted = min(adjusted, clip_threshold)
        if normalized_transform in ("log1p", "log1p_clip"):
            adjusted = float(np.log1p(adjusted))
        transformed.append(float(adjusted))
    diagnostics = {
        "transform": normalized_transform,
        "clip_percentile": float(clip_percentile),
        "clip_threshold": float(clip_threshold),
        "raw_max": raw_max,
        "raw_p95": raw_p95,
        "transformed_max": float(max(transformed) if transformed else 0.0),
        "candidate_count": float(len(clean_values)),
    }
    return transformed, diagnostics


def _transform_stage2_score_pairs(
    score_pairs: List[Tuple[int, float]],
    transform: str = "log1p_clip",
    clip_percentile: float = 95.0,
    clip_min_candidates: int = 8,
) -> Tuple[List[Tuple[int, float]], Dict[str, float]]:
    if not score_pairs:
        return [], {
            "transform": str(transform),
            "clip_percentile": float(clip_percentile),
            "clip_threshold": 0.0,
            "raw_max": 0.0,
            "raw_p95": 0.0,
            "transformed_max": 0.0,
            "candidate_count": 0.0,
        }
    pod_ids = [int(pid) for pid, _ in score_pairs]
    raw_values = [float(score) for _, score in score_pairs]
    transformed_values, diagnostics = _transform_stage2_score_values(
        raw_values,
        transform=transform,
        clip_percentile=clip_percentile,
        clip_min_candidates=clip_min_candidates,
    )
    transformed_pairs = sorted(
        zip(pod_ids, transformed_values),
        key=lambda item: item[1],
        reverse=True,
    )
    return transformed_pairs, diagnostics


def _aggregate_candidate_records(
    per_sample_results: List[Dict],
    total_nodes: int,
    top_k: int,
    vote_weight: float,
    mean_weight: float,
    peak_weight: float,
) -> List[int]:
    cand_sum: Dict[int, float] = defaultdict(float)
    cand_cnt: Dict[int, int] = defaultdict(int)
    cand_max: Dict[int, float] = defaultdict(lambda: float("-inf"))
    focus_sum: Dict[int, float] = defaultdict(float)
    focus_cnt: Dict[int, int] = defaultdict(int)
    focus_peak: Dict[int, float] = defaultdict(lambda: float("-inf"))
    focus_samples = 0
    focus_types = {"stress", "network", "podfail"}

    def _focus_family(ft: str) -> Optional[str]:
        ft = str(ft or "").strip().lower()
        if not ft:
            return None
        if "stress" in ft or "cpu" in ft or "memory" in ft:
            return "stress"
        if ft in {"partition", "loss", "bandwidth"} or "network" in ft:
            return "network"
        if ft in {"pod-failure", "container-kill"} or "kill" in ft or "oom" in ft:
            return "podfail"
        return None

    for rec in per_sample_results:
        rec_scores = rec.get("scores") or []
        rec_cands = rec.get("candidate_services") or []
        if not isinstance(rec_scores, list) or not rec_scores:
            continue
        for cid in rec_cands:
            try:
                idx = int(cid)
            except Exception:
                continue
            if 0 <= idx < len(rec_scores):
                score = float(rec_scores[idx])
                cand_sum[idx] += score
                cand_cnt[idx] += 1
                cand_max[idx] = max(cand_max[idx], score)
        focus_family = _focus_family(rec.get("failure_type", ""))
        if focus_family not in focus_types:
            continue
        focus_samples += 1
        focus_scores = rec.get("channel_b_scores") or rec_scores
        focus_cands = rec.get("channel_b_candidates") or rec_cands
        if not isinstance(focus_scores, list) or not focus_scores:
            continue
        for cid in focus_cands:
            try:
                idx = int(cid)
            except Exception:
                continue
            if 0 <= idx < len(focus_scores):
                score = float(focus_scores[idx])
                focus_sum[idx] += score
                focus_cnt[idx] += 1
                focus_peak[idx] = max(focus_peak[idx], score)
    if not cand_cnt:
        return []
    total_samples = max(len(per_sample_results), 1)
    max_count = max(cand_cnt.values()) if cand_cnt else 1
    candidate_ids = list(cand_cnt.keys())
    vote_scores = []
    mean_scores = []
    peak_scores = []
    focus_vote_scores = []
    focus_mean_scores = []
    focus_peak_scores = []
    for cid in candidate_ids:
        cnt = cand_cnt.get(cid, 0)
        vote_scores.append(float(cnt) / max(total_samples, 1))
        mean_scores.append(float(cand_sum.get(cid, 0.0)) / max(cnt, 1))
        peak_scores.append(float(cand_max.get(cid, 0.0)))
        fcnt = focus_cnt.get(cid, 0)
        focus_vote_scores.append(
            float(fcnt) / max(focus_samples, 1) if focus_samples > 0 else 0.0
        )
        focus_mean_scores.append(
            float(focus_sum.get(cid, 0.0)) / max(fcnt, 1) if fcnt > 0 else 0.0
        )
        focus_peak_scores.append(float(focus_peak.get(cid, 0.0)) if fcnt > 0 else 0.0)
    fused = _combine_score_tensors(
        [
            ("vote", th.tensor(vote_scores, dtype=th.float32), vote_weight),
            ("mean", th.tensor(mean_scores, dtype=th.float32), mean_weight),
            ("peak", th.tensor(peak_scores, dtype=th.float32), peak_weight),
            ("focus_vote", th.tensor(focus_vote_scores, dtype=th.float32), 0.10),
            ("focus_mean", th.tensor(focus_mean_scores, dtype=th.float32), 0.08),
            ("focus_peak", th.tensor(focus_peak_scores, dtype=th.float32), 0.04),
        ],
        device="cpu",
    )
    if fused is None:
        return sorted(candidate_ids, key=lambda cid: cand_cnt[cid], reverse=True)[
            :top_k
        ]
    ranking = sorted(
        zip(candidate_ids, fused.tolist(), vote_scores, peak_scores),
        key=lambda item: (item[1], item[2], item[3]),
        reverse=True,
    )
    return [cid for cid, _, _, _ in ranking[: min(top_k, total_nodes)]]


class LagAwareAttentionLayer(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, tau_max: int, num_heads: int = 1):
        super().__init__()
        self.tau_max = tau_max
        self.num_heads = num_heads
        self.out_feats = out_feats
        self.attn_linear = nn.Linear(in_feats * 2, num_heads)
        self.beta = nn.Parameter(th.tensor(1.0))
        self.proj = nn.Linear(in_feats, out_feats)

    def forward(
        self,
        H_v_0: th.Tensor,
        H_u_lags: Dict[int, th.Tensor],
        P_prior: Optional[Dict[int, float]] = None,
    ) -> Tuple[th.Tensor, Dict[int, float]]:
        if not H_u_lags:
            return self.proj(H_v_0), {0: 1.0}
        attn_scores = []
        lags = sorted(H_u_lags.keys())
        for lag in lags:
            H_u_k = H_u_lags[lag]
            if H_u_k.shape[0] == H_v_0.shape[0]:
                concat_feat = th.cat([H_v_0, H_u_k], dim=-1)
            else:
                if H_u_k.shape[0] > H_v_0.shape[0]:
                    H_u_k = H_u_k[: H_v_0.shape[0]]
                concat_feat = th.cat([H_v_0, H_u_k], dim=-1)
            score = self.attn_linear(concat_feat).mean(dim=0)
            if P_prior is not None and lag in P_prior:
                prior_score = self.beta * th.log(th.tensor(P_prior[lag] + 1e-10))
                score = score + prior_score
            attn_scores.append((lag, score))
        scores = th.stack([score for _, score in attn_scores], dim=0)
        alpha = F.softmax(scores, dim=0)
        h_agg = th.zeros_like(H_v_0)
        alpha_dist = {}
        for idx, (lag, _) in enumerate(attn_scores):
            weight = alpha[idx].mean().item()
            alpha_dist[lag] = weight
            h_agg = h_agg + weight * H_u_lags[lag][: h_agg.shape[0]]
        h_v_agg = self.proj(h_agg)
        tau_star_lag = max(alpha_dist.items(), key=lambda x: x[1])[0]
        return h_v_agg, alpha_dist


def _aggregate_prior_by_lag(
    P_prior: Optional[Dict[Tuple[int, int], Dict[int, float]]],
) -> Optional[Dict[int, float]]:
    if not P_prior:
        return None
    lag_probs: Dict[int, float] = defaultdict(float)
    for _edge, lag_dict in P_prior.items():
        if not isinstance(lag_dict, dict):
            continue
        for lag, prob in lag_dict.items():
            try:
                lag_i = int(lag)
                prob_f = float(prob)
            except Exception:
                continue
            lag_probs[lag_i] += prob_f
    total = float(sum(lag_probs.values()))
    if total <= 0:
        return None
    return {lag: (p / total) for lag, p in lag_probs.items()}


class LagAwareGNN(nn.Module):
    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        tau_max: int,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.tau_max = tau_max
        self.layers = nn.ModuleList()
        self.layers.append(
            LagAwareAttentionLayer(in_feats, hidden_feats, tau_max, num_heads)
        )
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(hidden_feats, hidden_feats, num_heads))
        self.output_proj = nn.Linear(num_heads * hidden_feats, out_feats)

    def forward(
        self,
        g: DGLGraph,
        H: Dict[str, Dict[int, th.Tensor]],
        P_prior_dict: Dict[Tuple[int, int], Dict[int, float]],
    ) -> Tuple[Dict[str, th.Tensor], Dict[Tuple[int, int], int]]:
        ntype = None
        for candidate_ntype in ["pod", "api"]:
            if candidate_ntype in H:
                ntype = candidate_ntype
                break
        if ntype is None:
            ntype = list(H.keys())[0] if H else None
            if ntype is None:
                return {}, {}
        H_0 = H[ntype][0]
        x = H_0
        prior_by_lag = _aggregate_prior_by_lag(P_prior_dict)
        dominant_lag: int = 0
        alpha_dist: Dict[int, float] = {}
        if len(self.layers) > 0:
            x, alpha_dist = self.layers[0](H_0, H[ntype], prior_by_lag)
            if alpha_dist:
                dominant_lag = int(max(alpha_dist.items(), key=lambda kv: kv[1])[0])
        for i in range(1, len(self.layers)):
            x = self.layers[i](g, x)
            if isinstance(x, tuple):
                x = x[0]
            if x.dim() == 3:
                x = x.view(x.shape[0], -1)
            else:
                x = x.view(x.shape[0], -1)
        node_repr = {ntype: self.output_proj(x)}
        tau_star: Dict[Tuple[int, int], int] = {}
        try:
            num_nodes = int(H_0.shape[0])
            for v in range(num_nodes):
                tau_star[(0, v)] = dominant_lag
        except Exception:
            tau_star = {}
        return node_repr, tau_star


def _rerank_shared_with_stage1(
    final_shared_ranking: List[Tuple[int, float]],
    stage1_per_sample: List[Dict],
    stage1_weight: float,
    stage2_weight: float,
    keep_top1_anchor: bool,
    anchor_margin: float,
) -> Tuple[List[Tuple[int, float]], List[float], List[float], List[int]]:
    if not final_shared_ranking or not stage1_per_sample:
        return final_shared_ranking, [], [], []
    topk_pod_ids = [pid for pid, _ in final_shared_ranking]
    stage2_vals = [float(s) for _, s in final_shared_ranking]
    stage1_vals: List[float] = []
    vote_counts: List[int] = []
    for pid in topk_pod_ids:
        votes: List[float] = []
        for rec in stage1_per_sample:
            rec_scores = rec.get("scores") or []
            if isinstance(rec_scores, list) and 0 <= int(pid) < len(rec_scores):
                try:
                    votes.append(float(rec_scores[int(pid)]))
                except Exception:
                    pass
        vote_counts.append(len(votes))
        stage1_vals.append(float(np.mean(votes)) if votes else 0.0)
    stage1_tensor = _safe_minmax_norm(th.tensor(stage1_vals, dtype=th.float32))
    stage2_tensor = _safe_minmax_norm(th.tensor(stage2_vals, dtype=th.float32))
    combined = _combine_score_tensors(
        [
            ("stage1", stage1_tensor, stage1_weight),
            ("stage2", stage2_tensor, stage2_weight),
        ],
        device="cpu",
    )
    if combined is None:
        return final_shared_ranking, stage1_vals, stage2_vals, vote_counts
    combined_list = [float(x) for x in combined.tolist()]
    if keep_top1_anchor and combined_list:
        combined_list[0] = max(combined_list) + anchor_margin
    sorted_idx = sorted(
        range(len(combined_list)), key=lambda i: combined_list[i], reverse=True
    )
    reranked = [final_shared_ranking[i] for i in sorted_idx]
    return reranked, stage1_vals, stage2_vals, vote_counts


def _build_per_sample_final_rankings(
    final_shared_ranking: List[Tuple[int, float]],
    all_stage2_scores: List[Tuple[int, float]],
    stage1_per_sample: List[Dict],
    top_k_pods: int,
    stage1_weight: float,
    stage2_weight: float,
    keep_top1_anchor: bool,
    anchor_margin: float,
) -> List[Dict]:
    if not final_shared_ranking and not all_stage2_scores:
        return []
    if not stage1_per_sample:
        return [{"final_ranking": final_shared_ranking[:top_k_pods]}]
    stage2_source = all_stage2_scores or final_shared_ranking
    stage2_score_map = {int(pid): float(score) for pid, score in stage2_source}
    shared_top_ids = [int(pid) for pid, _ in final_shared_ranking[:top_k_pods]]
    results: List[Dict] = []
    for rec in stage1_per_sample:
        stage1_scores = rec.get("scores") or []
        sample_candidates = rec.get("candidate_services") or []
        sample_pool: List[int] = []
        seen = set()
        for pid in sample_candidates:
            try:
                pid_i = int(pid)
            except Exception:
                continue
            if pid_i not in seen:
                sample_pool.append(pid_i)
                seen.add(pid_i)
        for pid_i in shared_top_ids:
            if pid_i not in seen:
                sample_pool.append(pid_i)
                seen.add(pid_i)
        if not sample_pool:
            results.append({"final_ranking": final_shared_ranking[:top_k_pods]})
            continue
        sample_stage1_vals: List[float] = []
        sample_stage2_vals: List[float] = []
        for pid in sample_pool:
            sample_stage1_vals.append(
                float(stage1_scores[pid]) if 0 <= pid < len(stage1_scores) else 0.0
            )
            sample_stage2_vals.append(float(stage2_score_map.get(pid, 0.0)))
        stage1_norm_tensor = _safe_minmax_norm(
            th.tensor(sample_stage1_vals, dtype=th.float32)
        )
        stage2_norm_tensor = _safe_minmax_norm(
            th.tensor(sample_stage2_vals, dtype=th.float32)
        )
        combined_tensor = _combine_score_tensors(
            [
                ("stage2", stage2_norm_tensor, stage2_weight),
                ("stage1", stage1_norm_tensor, stage1_weight),
            ],
            device="cpu",
        )
        if combined_tensor is None:
            final_ranking = [
                (pid, stage2_score_map.get(pid, 0.0))
                for pid in sample_pool[:top_k_pods]
            ]
            results.append({"final_ranking": final_ranking})
            continue
        combined = [float(x) for x in combined_tensor.tolist()]
        if combined and keep_top1_anchor:
            combined[0] = max(combined) + anchor_margin
        sorted_idx = sorted(
            range(len(combined)), key=lambda i: combined[i], reverse=True
        )
        final_ranking = [(sample_pool[i], combined[i]) for i in sorted_idx[:top_k_pods]]
        results.append({"final_ranking": final_ranking})
    return results


def _score_stage2_for_candidate_pool(
    candidate_ids: List[int],
    tau_star: Dict[Tuple[int, int], int],
    pod_feats: Dict[int, Dict[int, th.Tensor]],
    norm_pod_feats: Dict[int, th.Tensor],
    downstream_feats: Dict[int, th.Tensor],
    lambda_1: float,
    lambda_2: float,
    lag_alignment_enabled: bool = True,
    counterfactual_effect_enabled: bool = True,
    local_anomaly_score_enabled: bool = True,
    source_bonus_weight: float = 0.15,
    sink_penalty_weight: float = 0.40,
    source_first_weight: float = 0.30,
    return_diagnostics: bool = False,
) -> Union[
    List[Tuple[int, float]], Tuple[List[Tuple[int, float]], Dict[int, Dict[str, float]]]
]:
    counterfactual = LagCalibratedCounterfactual(None, tau_star)
    pod_scores: List[Tuple[int, float]] = []
    diagnostics: Dict[int, Dict[str, float]] = {}
    for service_id in candidate_ids:
        if service_id not in pod_feats:
            continue
        service_pods = pod_feats[service_id]
        service_norm = norm_pod_feats.get(service_id)
        if service_norm is None:
            continue
        if th.isnan(service_norm).any():
            service_norm = th.where(
                th.isnan(service_norm), th.zeros_like(service_norm), service_norm
            )
        service_tau_star = 0
        if lag_alignment_enabled:
            for (u, v), lag in tau_star.items():
                if v == service_id:
                    service_tau_star = lag
                    break
        for pod_id, pod_hist_feats in service_pods.items():
            pod_feat_tau = pod_hist_feats.get(
                service_tau_star, pod_hist_feats.get(0, service_norm)
            )
            if th.isnan(pod_feat_tau).any():
                pod_feat_tau = th.where(
                    th.isnan(pod_feat_tau), th.zeros_like(pod_feat_tau), pod_feat_tau
                )
            delta_Z_p = pod_multidimensional_anomaly_modeling(
                pod_feat_tau, service_tau_star, service_norm
            )
            if th.isnan(delta_Z_p).any():
                delta_Z_p = th.where(
                    th.isnan(delta_Z_p), th.zeros_like(delta_Z_p), delta_Z_p
                )
            delta_norm = float(delta_Z_p.norm().item())
            if np.isnan(delta_norm) or np.isinf(delta_norm):
                model_score = 0.0
            else:
                model_score = float(
                    th.sigmoid(th.tensor(delta_norm)).item() * delta_norm
                )
            if not local_anomaly_score_enabled:
                model_score = 0.0
            downstream_feat = downstream_feats.get(
                service_id, th.zeros_like(pod_feat_tau)
            )
            if th.isnan(downstream_feat).any():
                downstream_feat = th.where(
                    th.isnan(downstream_feat),
                    th.zeros_like(downstream_feat),
                    downstream_feat,
                )
            counterfactual_effect = counterfactual.counterfactual_effect(
                pod_id,
                service_id,
                pod_hist_feats,
                service_norm,
                downstream_feat,
                service_tau_star,
            )
            if np.isnan(counterfactual_effect) or np.isinf(counterfactual_effect):
                counterfactual_effect = 0.0
            if not counterfactual_effect_enabled:
                counterfactual_effect = 0.0
            self_anomaly = float((pod_feat_tau - service_norm).abs().norm().item())
            downstream_gap = float((downstream_feat - service_norm).abs().norm().item())
            source_bonus = max(self_anomaly - downstream_gap, 0.0)
            sink_penalty = max(downstream_gap - self_anomaly, 0.0)
            source_first_signal = _compute_source_first_signal(
                pod_hist_feats=pod_hist_feats,
                service_norm=service_norm,
                service_tau_star=service_tau_star,
            )
            cf_val = float(np.log1p(max(counterfactual_effect, 0.0)))
            source_bonus_val = float(np.log1p(max(source_bonus, 0.0)))
            sink_penalty_val = float(np.log1p(max(sink_penalty, 0.0)))
            source_first_val = float(source_first_signal)
            final_score = (
                lambda_1 * model_score
                + lambda_2 * cf_val
                + source_bonus_weight * source_bonus_val
                - sink_penalty_weight * sink_penalty_val
                + source_first_weight * source_first_val
            )
            if np.isnan(final_score) or np.isinf(final_score):
                final_score = 0.0
            pod_id_i = int(pod_id)
            pod_scores.append((pod_id_i, float(final_score)))
            diagnostics[pod_id_i] = {
                "service_id": int(service_id),
                "model_score_raw": float(model_score),
                "counterfactual_effect_raw": float(counterfactual_effect),
                "source_bonus_raw": float(source_bonus),
                "sink_penalty_raw": float(sink_penalty),
                "source_first_signal_raw": float(source_first_signal),
                "self_anomaly": float(self_anomaly),
                "downstream_gap": float(downstream_gap),
                "model_score": float(model_score),
                "counterfactual_effect": float(cf_val),
                "source_bonus": float(source_bonus_val),
                "sink_penalty": float(sink_penalty_val),
                "source_first_signal": float(source_first_val),
                "final_score_raw": float(final_score),
            }
    pod_scores.sort(key=lambda x: x[1], reverse=True)
    if return_diagnostics:
        return pod_scores, diagnostics
    return pod_scores


def _compute_top1_challenger_features(
    pod_id: int,
    candidate_service_ids: List[int],
    tau_star: Dict[Tuple[int, int], int],
    pod_feats: Dict[int, Dict[int, th.Tensor]],
    norm_pod_feats: Dict[int, th.Tensor],
    downstream_feats: Dict[int, th.Tensor],
    sample_scores: List[float],
) -> Dict[str, float]:
    owner_service_id: Optional[int] = None
    for service_id in candidate_service_ids:
        if pod_id in pod_feats.get(service_id, {}):
            owner_service_id = int(service_id)
            break
    if owner_service_id is None:
        for service_id, service_pods in pod_feats.items():
            if pod_id in service_pods:
                owner_service_id = int(service_id)
                break
    if owner_service_id is None:
        return {
            "source_margin": 0.0,
            "local_anomaly": 0.0,
            "lag_consistency": 0.0,
            "stage1_score": (
                float(sample_scores[pod_id])
                if 0 <= pod_id < len(sample_scores)
                else 0.0
            ),
        }
    service_norm = norm_pod_feats.get(owner_service_id)
    pod_hist_feats = pod_feats.get(owner_service_id, {}).get(pod_id)
    if service_norm is None or pod_hist_feats is None:
        return {
            "source_margin": 0.0,
            "local_anomaly": 0.0,
            "lag_consistency": 0.0,
            "stage1_score": (
                float(sample_scores[pod_id])
                if 0 <= pod_id < len(sample_scores)
                else 0.0
            ),
        }
    service_tau_star = 1
    for (u, v), lag in tau_star.items():
        if v == owner_service_id:
            service_tau_star = int(lag)
            break
    pod_feat_tau = pod_hist_feats.get(
        service_tau_star, pod_hist_feats.get(0, service_norm)
    )
    pod_feat_t0 = pod_hist_feats.get(0, pod_feat_tau)
    downstream_feat = downstream_feats.get(
        owner_service_id, th.zeros_like(pod_feat_tau)
    )
    if th.isnan(service_norm).any():
        service_norm = th.where(
            th.isnan(service_norm), th.zeros_like(service_norm), service_norm
        )
    if th.isnan(pod_feat_tau).any():
        pod_feat_tau = th.where(
            th.isnan(pod_feat_tau), th.zeros_like(pod_feat_tau), pod_feat_tau
        )
    if th.isnan(pod_feat_t0).any():
        pod_feat_t0 = th.where(
            th.isnan(pod_feat_t0), th.zeros_like(pod_feat_t0), pod_feat_t0
        )
    if th.isnan(downstream_feat).any():
        downstream_feat = th.where(
            th.isnan(downstream_feat), th.zeros_like(downstream_feat), downstream_feat
        )
    local_anomaly = float((pod_feat_tau - service_norm).abs().norm().item())
    downstream_gap = float((downstream_feat - service_norm).abs().norm().item())
    source_margin = local_anomaly - downstream_gap
    tau_anomaly = float((pod_feat_tau - service_norm).abs().mean().item())
    t0_anomaly = float((pod_feat_t0 - service_norm).abs().mean().item())
    lag_consistency = max(tau_anomaly - t0_anomaly, 0.0)
    return {
        "source_margin": float(source_margin),
        "local_anomaly": float(local_anomaly),
        "lag_consistency": float(lag_consistency),
        "stage1_score": (
            float(sample_scores[pod_id]) if 0 <= pod_id < len(sample_scores) else 0.0
        ),
    }


def _compute_source_first_signal(
    pod_hist_feats: Dict[int, th.Tensor],
    service_norm: th.Tensor,
    service_tau_star: int,
) -> float:
    lag_items = sorted((int(lag), feat) for lag, feat in pod_hist_feats.items())
    if not lag_items:
        return 0.0
    lag_scores: List[Tuple[int, float]] = []
    for lag, feat in lag_items:
        cur = feat
        if th.isnan(cur).any():
            cur = th.where(th.isnan(cur), th.zeros_like(cur), cur)
        anomaly = float((cur - service_norm).abs().norm().item())
        if np.isnan(anomaly) or np.isinf(anomaly):
            anomaly = 0.0
        lag_scores.append((lag, anomaly))
    max_anomaly = max(score for _, score in lag_scores)
    if max_anomaly <= 1e-8:
        return 0.0
    current_anomaly = next((score for lag, score in lag_scores if lag == 0), 0.0)
    significant_lags = [lag for lag, score in lag_scores if score >= 0.8 * max_anomaly]
    earliest_sig_lag = (
        max(significant_lags) if significant_lags else max(lag for lag, _ in lag_scores)
    )
    peak_lag = max(lag_scores, key=lambda item: item[1])[0]
    earlier_than_current = max(float(earliest_sig_lag), 0.0)
    peak_alignment = max(float(peak_lag - max(service_tau_star - 1, 0)), 0.0)
    current_sink_bias = max(current_anomaly / (max_anomaly + 1e-8) - 0.65, 0.0)
    return max(
        0.55 * earlier_than_current + 0.45 * peak_alignment - 0.75 * current_sink_bias,
        0.0,
    )


def _build_per_sample_final_rankings_true_stage2(
    stage1_per_sample: List[Dict],
    tau_star: Dict[Tuple[int, int], int],
    pod_feats: Dict[int, Dict[int, th.Tensor]],
    norm_pod_feats: Dict[int, th.Tensor],
    downstream_feats: Dict[int, th.Tensor],
    pcmci_results: Dict,
    top_k_pods: int,
    stage1_weight: float,
    stage2_weight: float,
    lambda_1: float,
    lambda_2: float,
    lag_alignment_enabled: bool = True,
    counterfactual_effect_enabled: bool = True,
    local_anomaly_score_enabled: bool = True,
    source_bonus_weight: float = 0.15,
    sink_penalty_weight: float = 0.40,
    source_first_weight: float = 0.30,
    shared_stage2_fallback: Optional[List[Tuple[int, float]]] = None,
) -> List[Dict]:
    results: List[Dict] = []
    fallback = shared_stage2_fallback or []
    for rec in stage1_per_sample:
        sample_candidates = []
        for pid in rec.get("candidate_services") or []:
            try:
                sample_candidates.append(int(pid))
            except Exception:
                continue
        sample_candidates = list(dict.fromkeys(sample_candidates))
        sample_scores = rec.get("scores") or []
        sample_stage2 = _score_stage2_for_candidate_pool(
            candidate_ids=sample_candidates,
            tau_star=tau_star,
            pod_feats=pod_feats,
            norm_pod_feats=norm_pod_feats,
            downstream_feats=downstream_feats,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lag_alignment_enabled=lag_alignment_enabled,
            counterfactual_effect_enabled=counterfactual_effect_enabled,
            local_anomaly_score_enabled=local_anomaly_score_enabled,
            source_bonus_weight=source_bonus_weight,
            sink_penalty_weight=sink_penalty_weight,
            source_first_weight=source_first_weight,
            return_diagnostics=True,
        )
        sample_stage2, sample_stage2_diag = sample_stage2
        sample_stage2 = node_confounder_pruning(sample_stage2, pcmci_results)
        stage2_map = {int(pid): float(score) for pid, score in sample_stage2}
        primary_pool: List[int] = []
        seen = set()
        for pid, _ in sample_stage2:
            pid_i = int(pid)
            if pid_i not in seen:
                primary_pool.append(pid_i)
                seen.add(pid_i)
        for pid in sample_candidates:
            if pid not in seen:
                primary_pool.append(pid)
                seen.add(pid)
        fallback_pool: List[int] = []
        for pid, _ in fallback[:top_k_pods]:
            pid_i = int(pid)
            if pid_i not in seen:
                fallback_pool.append(pid_i)
                seen.add(pid_i)
        if not primary_pool and not fallback_pool:
            results.append(
                {
                    "final_ranking": [],
                    "diagnostics": {
                        "sample_candidate_services": sample_candidates,
                        "sample_stage2_scores": [],
                        "sample_stage2_diagnostics": {},
                        "primary_pool": [],
                        "fallback_pool": [],
                        "shared_top1_fallback": fallback[0] if fallback else None,
                    },
                }
            )
            continue
        primary_ranking: List[Tuple[int, float]] = []
        if primary_pool:
            s1_vals = [
                float(sample_scores[pid]) if 0 <= pid < len(sample_scores) else 0.0
                for pid in primary_pool
            ]
            s2_vals = [float(stage2_map.get(pid, 0.0)) for pid in primary_pool]
            s1_tensor = _safe_minmax_norm(th.tensor(s1_vals, dtype=th.float32))
            s2_tensor = _safe_minmax_norm(th.tensor(s2_vals, dtype=th.float32))
            combined_tensor = _combine_score_tensors(
                [
                    ("stage1", s1_tensor, stage1_weight),
                    ("stage2", s2_tensor, stage2_weight),
                ],
                device="cpu",
            )
            if combined_tensor is None:
                primary_ranking = [
                    (pid, stage2_map.get(pid, 0.0)) for pid in primary_pool
                ]
            else:
                combined = [float(x) for x in combined_tensor.tolist()]
                primary_ranking = sorted(
                    zip(primary_pool, combined),
                    key=lambda item: item[1],
                    reverse=True,
                )
        fallback_ranking: List[Tuple[int, float]] = []
        if fallback_pool:
            fallback_scores = [
                float(score) * 0.85
                for _, score in fallback[:top_k_pods]
                if int(_) in fallback_pool
            ]
            fallback_map = {}
            for pid, score in fallback[:top_k_pods]:
                pid_i = int(pid)
                if pid_i in fallback_pool:
                    fallback_map[pid_i] = float(score) * 0.85
            fallback_ranking = sorted(
                [(pid, fallback_map.get(pid, 0.0)) for pid in fallback_pool],
                key=lambda item: item[1],
                reverse=True,
            )
        final_ranking = (primary_ranking + fallback_ranking)[:top_k_pods]
        top_stage2_diag: Dict[str, Dict[str, float]] = {}
        for pid, _ in sample_stage2[:top_k_pods]:
            pid_i = int(pid)
            top_stage2_diag[str(pid_i)] = sample_stage2_diag.get(pid_i, {})
        results.append(
            {
                "final_ranking": final_ranking,
                "diagnostics": {
                    "sample_candidate_services": sample_candidates,
                    "sample_stage2_scores": sample_stage2[:top_k_pods],
                    "sample_stage2_diagnostics": top_stage2_diag,
                    "primary_pool": primary_pool[:top_k_pods],
                    "fallback_pool": fallback_pool[:top_k_pods],
                    "shared_top1_fallback": fallback[0] if fallback else None,
                },
            }
        )
    return results


def pod_multidimensional_anomaly_modeling(
    pod_feats: th.Tensor,
    tau_star: int,
    norm_feats: th.Tensor,
) -> th.Tensor:
    if th.isnan(pod_feats).any():
        pod_feats = th.where(th.isnan(pod_feats), th.zeros_like(pod_feats), pod_feats)
    if th.isnan(norm_feats).any():
        norm_feats = th.where(
            th.isnan(norm_feats), th.zeros_like(norm_feats), norm_feats
        )
    delta = pod_feats - norm_feats
    if th.isnan(delta).any():
        delta = th.where(th.isnan(delta), th.zeros_like(delta), delta)
    return delta


class LagCalibratedCounterfactual:
    def __init__(
        self,
        predictor_model: nn.Module,
        tau_star: Dict[Tuple[int, int], int],
    ):
        self.predictor = predictor_model
        self.tau_star = tau_star

    def counterfactual_effect(
        self,
        pod_id: int,
        service_id: int,
        Z_p_hist: Dict[int, th.Tensor],
        Z_p_norm: th.Tensor,
        downstream_feats: th.Tensor,
        tau_star_lag: int,
    ) -> float:
        if tau_star_lag in Z_p_hist:
            Z_p_factual = Z_p_hist[tau_star_lag]
        else:
            Z_p_factual = Z_p_hist.get(0, Z_p_norm)
        if th.isnan(Z_p_factual).any():
            Z_p_factual = th.where(
                th.isnan(Z_p_factual), th.zeros_like(Z_p_factual), Z_p_factual
            )
        if th.isnan(Z_p_norm).any():
            Z_p_norm = th.where(th.isnan(Z_p_norm), th.zeros_like(Z_p_norm), Z_p_norm)
        Z_p_intervened = Z_p_norm
        diff = Z_p_factual - Z_p_intervened
        if th.isnan(diff).any():
            diff = th.where(th.isnan(diff), th.zeros_like(diff), diff)
        effect = _stable_rms_norm(diff)
        if np.isnan(effect) or np.isinf(effect):
            effect = 0.0
        return effect


def stage2_fine_grained_localization(
    S_cand: List[int],
    tau_star: Dict[Tuple[int, int], int],
    pod_feats: Dict[int, Dict[int, th.Tensor]],
    norm_pod_feats: Dict[int, th.Tensor],
    downstream_feats: Dict[int, th.Tensor],
    predictor_model: Optional[nn.Module] = None,
    lambda_1: float = 0.6,
    lambda_2: float = 0.4,
    lag_alignment_enabled: bool = True,
    counterfactual_effect_enabled: bool = True,
    local_anomaly_score_enabled: bool = True,
    verbose: bool = True,
) -> List[Tuple[int, float]]:
    logger.info("Stage 2: 开始细粒度定位...")
    counterfactual = LagCalibratedCounterfactual(predictor_model, tau_star)
    pod_scores = []
    for service_id in S_cand:
        if service_id not in pod_feats:
            continue
        service_pods = pod_feats[service_id]
        service_norm = norm_pod_feats.get(service_id, None)
        if service_norm is None:
            continue
        service_tau_star = 0
        if lag_alignment_enabled:
            for (u, v), lag in tau_star.items():
                if v == service_id:
                    service_tau_star = lag
                    break
        for pod_id, pod_hist_feats in service_pods.items():
            if service_tau_star in pod_hist_feats:
                pod_feat_tau = pod_hist_feats[service_tau_star]
            else:
                pod_feat_tau = pod_hist_feats.get(0, service_norm)
            if th.isnan(pod_feat_tau).any():
                pod_feat_tau = th.where(
                    th.isnan(pod_feat_tau), th.zeros_like(pod_feat_tau), pod_feat_tau
                )
            if th.isnan(service_norm).any():
                service_norm = th.where(
                    th.isnan(service_norm), th.zeros_like(service_norm), service_norm
                )
            delta_Z_p = pod_multidimensional_anomaly_modeling(
                pod_feat_tau, service_tau_star, service_norm
            )
            if th.isnan(delta_Z_p).any():
                delta_Z_p = th.where(
                    th.isnan(delta_Z_p), th.zeros_like(delta_Z_p), delta_Z_p
                )
            delta_norm = _stable_rms_norm(delta_Z_p)
            if np.isnan(delta_norm) or np.isinf(delta_norm):
                model_score = 0.0
            else:
                model_score = float(
                    np.log1p(th.sigmoid(th.tensor(delta_norm)).item() * delta_norm)
                )
            if not local_anomaly_score_enabled:
                model_score = 0.0
            downstream_feat = downstream_feats.get(
                service_id, th.zeros_like(pod_feat_tau)
            )
            if th.isnan(downstream_feat).any():
                downstream_feat = th.where(
                    th.isnan(downstream_feat),
                    th.zeros_like(downstream_feat),
                    downstream_feat,
                )
            counterfactual_effect = counterfactual.counterfactual_effect(
                pod_id,
                service_id,
                pod_hist_feats,
                service_norm,
                downstream_feat,
                service_tau_star,
            )
            if np.isnan(counterfactual_effect) or np.isinf(counterfactual_effect):
                counterfactual_effect = 0.0
            counterfactual_effect = float(np.log1p(max(counterfactual_effect, 0.0)))
            if not counterfactual_effect_enabled:
                counterfactual_effect = 0.0
            final_score = lambda_1 * model_score + lambda_2 * counterfactual_effect
            if np.isnan(final_score) or np.isinf(final_score):
                final_score = 0.0
            pod_scores.append((pod_id, final_score))
    pod_scores.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Stage 2: 完成，评估了 {len(pod_scores)} 个Pod")
    return pod_scores


def node_confounder_pruning(
    pod_scores: List[Tuple[int, float]],
    pcmci_results: Dict,
    threshold: float = 0.1,
) -> List[Tuple[int, float]]:
    pruned_scores = []
    for pod_id, score in pod_scores:
        should_prune = False
        if should_prune:
            score = score * 0.5
        pruned_scores.append((pod_id, score))
    return sorted(pruned_scores, key=lambda x: x[1], reverse=True)


def _normalize_failure_type(label: Optional[Dict]) -> str:
    if not isinstance(label, dict):
        return "unknown"
    value = str(label.get("failure_type", "") or "").strip().lower()
    if not value:
        return "unknown"
    return value


def _channel_profile_from_failure_type(failure_type: str) -> Dict[str, float]:
    ft = failure_type.lower()
    profile = {
        "response_weight": 0.55,
        "dual_hit_bonus": 0.10,
        "delta_scale": 1.0,
        "peak_scale": 1.0,
        "prefer_delta": 0.50,
        "feature_slice": None,
        "aux_feature_slice": None,
        "aux_blend": 0.0,
    }
    if "delay" in ft:
        profile.update(
            {
                "response_weight": 0.70,
                "dual_hit_bonus": 0.18,
                "delta_scale": 1.20,
                "peak_scale": 1.10,
                "prefer_delta": 0.60,
            }
        )
    elif "abort" in ft or "error" in ft or "exception" in ft:
        profile.update(
            {
                "response_weight": 0.62,
                "dual_hit_bonus": 0.16,
                "delta_scale": 1.30,
                "peak_scale": 0.95,
                "prefer_delta": 0.65,
            }
        )
    elif "stress" in ft or "cpu" in ft or "memory" in ft:
        profile.update(
            {
                "response_weight": 0.62,
                "dual_hit_bonus": 0.18,
                "delta_scale": 1.00,
                "peak_scale": 1.20,
                "prefer_delta": 0.45,
                "aux_feature_slice": (0, 4),
                "aux_blend": 0.70,
            }
        )
    elif ft in {"partition", "loss", "bandwidth"} or "network" in ft:
        profile.update(
            {
                "response_weight": 0.60,
                "dual_hit_bonus": 0.18,
                "delta_scale": 1.10,
                "peak_scale": 1.15,
                "prefer_delta": 0.50,
                "aux_feature_slice": (4, 8),
                "aux_blend": 0.68,
            }
        )
    elif ft in {"pod-failure", "container-kill"} or "kill" in ft or "oom" in ft:
        profile.update(
            {
                "response_weight": 0.68,
                "dual_hit_bonus": 0.22,
                "delta_scale": 1.20,
                "peak_scale": 1.05,
                "prefer_delta": 0.58,
                "aux_feature_slice": (8, 12),
                "aux_blend": 0.75,
            }
        )
    elif "mysql" in ft or "db" in ft or "database" in ft:
        profile.update(
            {
                "response_weight": 0.63,
                "dual_hit_bonus": 0.20,
                "delta_scale": 1.10,
                "peak_scale": 1.05,
                "prefer_delta": 0.55,
                "aux_feature_slice": (12, 16),
                "aux_blend": 0.70,
            }
        )
    return profile


def _extract_temporal_response_scores_v2(
    H: Dict[str, Dict[int, th.Tensor]],
    ntype: str,
    profile: Dict[str, float],
    aux_feat: Optional[th.Tensor] = None,
    device: str = "cpu",
) -> Optional[th.Tensor]:
    if ntype not in H or 0 not in H[ntype]:
        return None
    current_feat = H[ntype][0]
    prev_feat = H[ntype].get(1, current_feat)
    feature_slice = profile.get("feature_slice")
    aux_feature_slice = profile.get("aux_feature_slice", feature_slice)
    focus_current = current_feat
    focus_prev = prev_feat
    if (
        isinstance(feature_slice, tuple)
        and len(feature_slice) == 2
        and int(feature_slice[1]) <= int(current_feat.shape[1])
    ):
        start, end = int(feature_slice[0]), int(feature_slice[1])
        focus_current = current_feat[:, start:end]
        focus_prev = prev_feat[:, start:end]
    current_norm = focus_current.norm(dim=1)
    delta_norm = (focus_current - focus_prev).abs().norm(dim=1) * float(
        profile.get("delta_scale", 1.0)
    )
    lag_energies: List[th.Tensor] = []
    for lag in sorted(H[ntype].keys()):
        lag_feat = H[ntype][lag]
        if (
            isinstance(feature_slice, tuple)
            and len(feature_slice) == 2
            and int(feature_slice[1]) <= int(lag_feat.shape[1])
        ):
            start, end = int(feature_slice[0]), int(feature_slice[1])
            lag_feat = lag_feat[:, start:end]
        lag_energies.append(lag_feat.abs().norm(dim=1))
    peak_norm = None
    if lag_energies:
        peak_norm = th.stack(lag_energies, dim=0).max(dim=0).values * float(
            profile.get("peak_scale", 1.0)
        )
    prefer_delta = float(profile.get("prefer_delta", 0.5))
    remain = max(1.0 - prefer_delta, 0.0)
    current_w = remain * 0.65
    peak_w = remain * 0.35
    base_scores = _combine_score_tensors(
        [
            ("current", current_norm, current_w),
            ("delta", delta_norm, prefer_delta),
            ("peak", peak_norm, peak_w),
        ],
        device=device,
    )
    if aux_feat is None or aux_feat.ndim != 2:
        return base_scores
    if (
        isinstance(aux_feature_slice, tuple)
        and len(aux_feature_slice) == 2
        and int(aux_feature_slice[1]) <= int(aux_feat.shape[1])
    ):
        start, end = int(aux_feature_slice[0]), int(aux_feature_slice[1])
        aux_scores = _safe_minmax_norm(aux_feat[:, start:end].norm(dim=1))
        aux_blend = float(profile.get("aux_blend", 0.70))
        base_blend = max(1.0 - aux_blend, 0.0)
        return _combine_score_tensors(
            [
                ("aux", aux_scores, aux_blend),
                ("base", base_scores, base_blend),
            ],
            device=device,
        )
    return base_scores


def _stage1_source_aware_profile(failure_type: str) -> Dict[str, float]:
    ft = str(failure_type or "").strip().lower()
    profile = {
        "mechanism": "generic",
        "base": 0.78,
        "source_first": 0.10,
        "local_margin": 0.10,
        "current_change": 0.06,
        "current_drop": 0.00,
        "sink_penalty": 0.06,
    }
    if any(key in ft for key in ("stress", "cpu", "memory")):
        profile.update(
            {
                "mechanism": "resource",
                "base": 0.62,
                "source_first": 0.14,
                "local_margin": 0.18,
                "current_change": 0.08,
                "current_drop": 0.00,
                "sink_penalty": 0.06,
            }
        )
    elif any(
        key in ft
        for key in ("container-kill", "pod-failure", "pod-kill", "kill", "oom")
    ):
        profile.update(
            {
                "mechanism": "availability",
                "base": 0.60,
                "source_first": 0.00,
                "local_margin": 0.20,
                "current_change": 0.18,
                "current_drop": 0.14,
                "sink_penalty": 0.00,
            }
        )
    elif any(key in ft for key in ("exception", "corrupt", "return")):
        profile.update(
            {
                "mechanism": "local_fault",
                "base": 0.72,
                "source_first": 0.06,
                "local_margin": 0.14,
                "current_change": 0.10,
                "current_drop": 0.00,
                "sink_penalty": 0.02,
            }
        )
    elif any(key in ft for key in ("request-delay", "response-delay", "delay")):
        profile.update(
            {
                "mechanism": "latency",
                "base": 0.72,
                "source_first": 0.08,
                "local_margin": 0.08,
                "current_change": 0.10,
                "current_drop": 0.00,
                "sink_penalty": 0.03,
            }
        )
    elif any(key in ft for key in ("bandwidth", "partition", "loss", "network")):
        profile.update(
            {
                "mechanism": "network",
                "base": 0.66,
                "source_first": 0.10,
                "local_margin": 0.16,
                "current_change": 0.08,
                "current_drop": 0.00,
                "sink_penalty": 0.04,
            }
        )
    elif any(key in ft for key in ("replace-code", "replace-method")):
        profile.update(
            {
                "mechanism": "http_replace",
                "base": 0.92,
                "source_first": 0.02,
                "local_margin": 0.04,
                "current_change": 0.03,
                "current_drop": 0.00,
                "sink_penalty": 0.00,
            }
        )
    return profile


def _compute_stage1_source_aware_terms(
    H: Dict[str, Dict[int, th.Tensor]],
    ntype: str,
    device: str = "cpu",
) -> Dict[str, Optional[th.Tensor]]:
    if ntype not in H or 0 not in H[ntype]:
        return {
            "source_first": None,
            "local_margin": None,
            "current_change": None,
            "current_drop": None,
            "sink_penalty": None,
        }
    current_feat = H[ntype][0]
    if th.isnan(current_feat).any():
        current_feat = th.where(
            th.isnan(current_feat), th.zeros_like(current_feat), current_feat
        )
    current_energy = current_feat.abs().norm(dim=1)
    current_abs_mean = current_feat.abs().mean(dim=1)
    lag_energies: List[Tuple[int, th.Tensor]] = []
    for lag, feat in sorted(H[ntype].items()):
        if th.isnan(feat).any():
            feat = th.where(th.isnan(feat), th.zeros_like(feat), feat)
        lag_energies.append((int(lag), feat.abs().norm(dim=1)))
    historical = [energy for lag, energy in lag_energies if lag > 0]
    if historical:
        hist_stack = th.stack(historical, dim=0)
        hist_peak = hist_stack.max(dim=0).values
        hist_mean = hist_stack.mean(dim=0)
    else:
        hist_peak = current_energy
        hist_mean = current_energy
    prev_energy = H[ntype].get(1, current_feat)
    if th.isnan(prev_energy).any():
        prev_energy = th.where(
            th.isnan(prev_energy), th.zeros_like(prev_energy), prev_energy
        )
    prev_abs_mean = prev_energy.abs().mean(dim=1)
    max_energy = th.maximum(current_energy, hist_peak) + 1e-8
    earlier_than_current = th.clamp((hist_peak - current_energy) / max_energy, min=0.0)
    sustained_history = th.clamp(hist_mean / max_energy - 0.35, min=0.0)
    source_first = _safe_minmax_norm(
        0.75 * earlier_than_current + 0.25 * sustained_history
    )
    local_center = (
        current_energy.mean()
        if current_energy.numel() > 0
        else th.tensor(0.0, device=current_energy.device)
    )
    local_margin = _safe_minmax_norm(th.clamp(current_energy - local_center, min=0.0))
    current_only_spike = th.clamp((current_energy - hist_peak) / max_energy, min=0.0)
    sink_penalty = _safe_minmax_norm(current_only_spike)
    current_change = _safe_minmax_norm((current_abs_mean - prev_abs_mean).abs())
    current_drop = _safe_minmax_norm(
        th.clamp(prev_abs_mean - current_abs_mean, min=0.0)
    )
    return {
        "source_first": source_first.to(device) if source_first is not None else None,
        "local_margin": local_margin.to(device) if local_margin is not None else None,
        "current_change": (
            current_change.to(device) if current_change is not None else None
        ),
        "current_drop": current_drop.to(device) if current_drop is not None else None,
        "sink_penalty": sink_penalty.to(device) if sink_penalty is not None else None,
    }


def stage1_fault_type_dual_channel_localization(
    graphs: List[DGLGraph],
    stacked_nfeat: Dict[str, th.Tensor],
    data_stats: Dict[str, Dict[str, th.Tensor]],
    labels: List[Dict],
    model: LagAwareGNN,
    tau_max: int,
    P_prior: Dict[Tuple[int, int], Dict[int, float]],
    device: str = "cpu",
    top_k: int = 10,
    ntype: Optional[str] = None,
    expand_factor: float = 2.0,
    gnn_weight: float = 0.45,
    anomaly_weight: float = 0.55,
    response_weight: float = 0.55,
    vote_weight: float = 0.45,
    mean_weight: float = 0.35,
    peak_weight: float = 0.20,
    dual_hit_bonus: float = 0.10,
    stage1_channel_aux: Optional[th.Tensor] = None,
    temporal_response_channel_enabled: bool = True,
    dual_channel_consistency_enabled: bool = True,
    source_aware_enabled: bool = False,
    source_aware_strength: float = 1.0,
    weak_type_conditional_expansion_enabled: bool = False,
    weak_type_expand_factor: float = 2.0,
    weak_failure_types: Optional[Iterable[str]] = None,
) -> Tuple[List[int], Dict[Tuple[int, int], int], List[Dict]]:
    logger.info("Stage 1 fault-type-aware dual-channel recall...")
    if ntype is None:
        for candidate_ntype in ["pod", "api"]:
            if candidate_ntype in stacked_nfeat:
                ntype = candidate_ntype
                break
        if ntype is None:
            ntype = list(stacked_nfeat.keys())[0] if stacked_nfeat else "api"
    tau_star_dict: Dict[Tuple[int, int], int] = {}
    per_sample_results: List[Dict] = []
    global_candidate_votes: List[int] = []
    total_nodes = 0
    base_expanded_top_k = max(top_k, int(round(top_k * max(expand_factor, 1.0))))
    weak_expanded_top_k = max(
        top_k, int(round(top_k * max(weak_type_expand_factor, 1.0)))
    )
    weak_type_set = {
        str(ft).strip().lower() for ft in (weak_failure_types or []) if str(ft).strip()
    }
    for i, g in enumerate(tqdm(graphs, desc="Stage 1 typed dual", total=len(labels))):
        label = labels[i] if i < len(labels) else {}
        failure_type = _normalize_failure_type(label)
        profile = _channel_profile_from_failure_type(failure_type)
        expanded_top_k = base_expanded_top_k
        if weak_type_conditional_expansion_enabled and failure_type in weak_type_set:
            expanded_top_k = weak_expanded_top_k
        aux_sample = None
        if (
            stage1_channel_aux is not None
            and stage1_channel_aux.ndim == 3
            and i < int(stage1_channel_aux.shape[0])
        ):
            aux_sample = stage1_channel_aux[i].to(device)
        H = historical_window_encoding(stacked_nfeat, data_stats, tau_max, i, device)
        node_repr, tau_star = model(g.to(device), H, P_prior)
        tau_star_dict.update(tau_star)
        channel_a_scores = None
        channel_b_scores = None
        union_scores = None
        top_a: List[int] = []
        top_b: List[int] = []
        union_candidates: List[int] = []
        source_aware_debug: Dict[str, float] = {"enabled": 0.0}
        if ntype in H and 0 in H[ntype]:
            current_feat = H[ntype][0]
            total_nodes = max(total_nodes, int(current_feat.shape[0]))
            fdim = int(current_feat.shape[-1])
            if fdim >= 20:
                anomaly_scores = current_feat[:, 10:20].norm(dim=1)
            elif fdim >= 10:
                anomaly_scores = current_feat[:, 2:10].norm(dim=1)
            else:
                anomaly_scores = current_feat.norm(dim=1)
            gnn_scores = node_repr[ntype].norm(dim=1) if ntype in node_repr else None
            channel_a_scores = _combine_score_tensors(
                [
                    ("gnn", gnn_scores, gnn_weight),
                    ("anomaly", anomaly_scores, anomaly_weight),
                ],
                device=device,
            )
            channel_b_scores = None
            if temporal_response_channel_enabled:
                channel_b_scores = _extract_temporal_response_scores_v2(
                    H,
                    ntype,
                    profile,
                    aux_feat=aux_sample,
                    device=device,
                )
            if (
                temporal_response_channel_enabled
                and channel_b_scores is not None
                and anomaly_scores is not None
            ):
                typed_response_weight = float(
                    profile.get("response_weight", response_weight)
                )
                channel_b_scores = _combine_score_tensors(
                    [
                        ("response", channel_b_scores, typed_response_weight),
                        (
                            "anomaly",
                            anomaly_scores,
                            max(1.0 - typed_response_weight, 0.0),
                        ),
                    ],
                    device=device,
                )
            if channel_a_scores is None:
                channel_a_scores = _safe_minmax_norm(gnn_scores)
            if temporal_response_channel_enabled and channel_b_scores is None:
                channel_b_scores = _safe_minmax_norm(anomaly_scores)
            k_a = (
                min(expanded_top_k, int(channel_a_scores.shape[0]))
                if channel_a_scores is not None
                else 0
            )
            k_b = (
                min(expanded_top_k, int(channel_b_scores.shape[0]))
                if temporal_response_channel_enabled and channel_b_scores is not None
                else 0
            )
            if k_a > 0:
                top_a = channel_a_scores.topk(k_a).indices.tolist()
            if k_b > 0:
                top_b = channel_b_scores.topk(k_b).indices.tolist()
            typed_bonus = (
                float(profile.get("dual_hit_bonus", dual_hit_bonus))
                if dual_channel_consistency_enabled
                else 0.0
            )
            hit_counts = defaultdict(int)
            for pid in top_a:
                hit_counts[int(pid)] += 1
            for pid in top_b:
                hit_counts[int(pid)] += 1
            union_candidate_ids = sorted(set(top_a) | set(top_b))
            union_values: List[float] = []
            for pid in union_candidate_ids:
                a_val = (
                    float(channel_a_scores[pid])
                    if channel_a_scores is not None and 0 <= pid < len(channel_a_scores)
                    else 0.0
                )
                b_val = (
                    float(channel_b_scores[pid])
                    if channel_b_scores is not None and 0 <= pid < len(channel_b_scores)
                    else 0.0
                )
                bonus = typed_bonus if hit_counts.get(int(pid), 0) >= 2 else 0.0
                union_values.append(max(a_val, b_val) + bonus)
            union_norm = (
                _safe_minmax_norm(th.tensor(union_values, dtype=th.float32))
                if union_values
                else None
            )
            if source_aware_enabled and union_values and union_norm is not None:
                source_profile = _stage1_source_aware_profile(failure_type)
                strength = max(float(source_aware_strength), 0.0)
                base_w = float(source_profile.get("base", 0.78))
                source_first_w = (
                    float(source_profile.get("source_first", 0.10)) * strength
                )
                local_margin_w = (
                    float(source_profile.get("local_margin", 0.10)) * strength
                )
                current_change_w = (
                    float(source_profile.get("current_change", 0.06)) * strength
                )
                current_drop_w = (
                    float(source_profile.get("current_drop", 0.0)) * strength
                )
                sink_penalty_w = (
                    float(source_profile.get("sink_penalty", 0.06)) * strength
                )
                source_terms = _compute_stage1_source_aware_terms(
                    H, ntype, device=device
                )

                def _gather_term(term: Optional[th.Tensor]) -> Optional[th.Tensor]:
                    if term is None:
                        return None
                    vals = []
                    for pid in union_candidate_ids:
                        if 0 <= int(pid) < int(term.shape[0]):
                            vals.append(float(term[int(pid)].item()))
                        else:
                            vals.append(0.0)
                    return th.tensor(vals, dtype=th.float32)

                source_first_vals = _gather_term(source_terms.get("source_first"))
                local_margin_vals = _gather_term(source_terms.get("local_margin"))
                current_change_vals = _gather_term(source_terms.get("current_change"))
                current_drop_vals = _gather_term(source_terms.get("current_drop"))
                sink_penalty_vals = _gather_term(source_terms.get("sink_penalty"))
                shaped = base_w * union_norm
                denom = max(base_w, 0.0)
                if source_first_vals is not None and source_first_w > 0:
                    sf_norm = _safe_minmax_norm(source_first_vals)
                    if sf_norm is not None:
                        shaped = shaped + source_first_w * sf_norm
                        denom += source_first_w
                if local_margin_vals is not None and local_margin_w > 0:
                    lm_norm = _safe_minmax_norm(local_margin_vals)
                    if lm_norm is not None:
                        shaped = shaped + local_margin_w * lm_norm
                        denom += local_margin_w
                if current_change_vals is not None and current_change_w > 0:
                    cc_norm = _safe_minmax_norm(current_change_vals)
                    if cc_norm is not None:
                        shaped = shaped + current_change_w * cc_norm
                        denom += current_change_w
                if current_drop_vals is not None and current_drop_w > 0:
                    cd_norm = _safe_minmax_norm(current_drop_vals)
                    if cd_norm is not None:
                        shaped = shaped + current_drop_w * cd_norm
                        denom += current_drop_w
                if sink_penalty_vals is not None and sink_penalty_w > 0:
                    sink_norm = _safe_minmax_norm(sink_penalty_vals)
                    if sink_norm is not None:
                        shaped = shaped - sink_penalty_w * sink_norm
                        denom += sink_penalty_w
                if denom > 0:
                    shaped = shaped / denom
                shaped_norm = _safe_minmax_norm(shaped) if shaped is not None else None
                if shaped_norm is not None:
                    union_norm = shaped_norm
                    union_values = [float(x) for x in union_norm.tolist()]
                    source_aware_debug = {
                        "enabled": 1.0,
                        "mechanism": str(source_profile.get("mechanism", "generic")),
                        "base_weight": base_w,
                        "source_first_weight": source_first_w,
                        "local_margin_weight": local_margin_w,
                        "current_change_weight": current_change_w,
                        "current_drop_weight": current_drop_w,
                        "sink_penalty_weight": sink_penalty_w,
                    }
            union_scores = th.zeros(
                total_nodes if total_nodes > 0 else len(union_candidate_ids),
                dtype=th.float32,
            )
            if union_norm is not None:
                for local_idx, pid in enumerate(union_candidate_ids):
                    if pid >= union_scores.shape[0]:
                        pad = th.zeros(
                            pid - union_scores.shape[0] + 1, dtype=th.float32
                        )
                        union_scores = th.cat([union_scores, pad], dim=0)
                    union_scores[pid] = union_norm[local_idx]
            union_candidates = [
                pid
                for _, pid in sorted(
                    zip(union_values, union_candidate_ids), reverse=True
                )
            ][:expanded_top_k]
            global_candidate_votes.extend(union_candidates)
        per_sample_results.append(
            {
                "index": i,
                "failure_type": failure_type,
                "candidate_services": union_candidates,
                "scores": _tensor_to_score_list(union_scores),
                "channel_a_scores": _tensor_to_score_list(channel_a_scores),
                "channel_b_scores": _tensor_to_score_list(channel_b_scores),
                "channel_a_candidates": top_a,
                "channel_b_candidates": top_b,
                "source_aware_score": source_aware_debug,
                "conditional_expansion": {
                    "enabled": bool(weak_type_conditional_expansion_enabled),
                    "applied": bool(
                        weak_type_conditional_expansion_enabled
                        and failure_type in weak_type_set
                    ),
                    "expanded_top_k": int(expanded_top_k),
                },
            }
        )
    S_cand = _aggregate_candidate_records(
        per_sample_results=per_sample_results,
        total_nodes=max(total_nodes, len(set(global_candidate_votes))),
        top_k=top_k,
        vote_weight=vote_weight,
        mean_weight=mean_weight,
        peak_weight=peak_weight,
    )
    if not S_cand:
        candidate_counts = {
            c: global_candidate_votes.count(c) for c in set(global_candidate_votes)
        }
        S_cand = sorted(
            candidate_counts.keys(), key=lambda x: candidate_counts[x], reverse=True
        )[:top_k]
    logger.info(
        f"Stage 1 fault-type-aware recall completed, retained {len(S_cand)} candidates"
    )
    return S_cand, tau_star_dict, per_sample_results


def _compute_relative_source_signal(
    candidate_pool: List[Tuple[int, float]],
    tau_star: Dict[Tuple[int, int], int],
    pod_feats: Dict[int, Dict[int, Dict[int, th.Tensor]]],
    norm_pod_feats: Dict[int, th.Tensor],
    downstream_feats: Dict[int, th.Tensor],
    lag_alignment_enabled: bool = True,
) -> Tuple[Dict[int, float], Dict[int, Dict[str, float]]]:
    candidate_ids = [int(pid) for pid, _ in candidate_pool]
    base_self: Dict[int, float] = {}
    base_delta: Dict[int, float] = {}
    base_downstream: Dict[int, float] = {}
    for pid in candidate_ids:
        service_id = pid if pid in pod_feats else None
        if service_id is None:
            base_self[pid] = 0.0
            base_delta[pid] = 0.0
            base_downstream[pid] = 0.0
            continue
        service_pods = pod_feats.get(service_id, {})
        pod_hist_feats = service_pods.get(pid, {})
        service_norm = norm_pod_feats.get(service_id)
        downstream_feat = downstream_feats.get(service_id)
        if service_norm is None or downstream_feat is None or not pod_hist_feats:
            base_self[pid] = 0.0
            base_delta[pid] = 0.0
            base_downstream[pid] = 0.0
            continue
        service_tau = 0
        if lag_alignment_enabled:
            for (u, v), lag in tau_star.items():
                if v == service_id:
                    service_tau = lag
                    break
        current_feat = pod_hist_feats.get(
            service_tau, pod_hist_feats.get(0, service_norm)
        )
        prev_feat = pod_hist_feats.get(
            min(service_tau + 1, max(pod_hist_feats.keys())), service_norm
        )
        base_self[pid] = float((current_feat - service_norm).abs().norm().item())
        base_delta[pid] = float((current_feat - prev_feat).abs().norm().item())
        base_downstream[pid] = float(
            (downstream_feat - service_norm).abs().norm().item()
        )
    raw_scores: Dict[int, float] = {}
    diagnostics: Dict[int, Dict[str, float]] = {}
    for pid in candidate_ids:
        others = [oid for oid in candidate_ids if oid != pid]
        if others:
            other_self_mean = float(
                np.mean([base_self.get(oid, 0.0) for oid in others])
            )
            other_delta_mean = float(
                np.mean([base_delta.get(oid, 0.0) for oid in others])
            )
        else:
            other_self_mean = 0.0
            other_delta_mean = 0.0
        self_advantage = max(base_self.get(pid, 0.0) - other_self_mean, 0.0)
        delta_advantage = max(base_delta.get(pid, 0.0) - other_delta_mean, 0.0)
        downstream_margin = max(
            base_self.get(pid, 0.0) - base_downstream.get(pid, 0.0), 0.0
        )
        dominance = self_advantage + 0.75 * delta_advantage + 0.75 * downstream_margin
        raw_scores[pid] = dominance
        diagnostics[pid] = {
            "self_advantage": self_advantage,
            "delta_advantage": delta_advantage,
            "downstream_margin": downstream_margin,
            "self_raw": base_self.get(pid, 0.0),
            "delta_raw": base_delta.get(pid, 0.0),
            "downstream_raw": base_downstream.get(pid, 0.0),
        }
    norm_tensor = _safe_minmax_norm(
        th.tensor([raw_scores[pid] for pid in candidate_ids], dtype=th.float32)
    )
    if norm_tensor is None:
        return {pid: 0.0 for pid in candidate_ids}, diagnostics
    return {
        pid: float(norm_tensor[idx]) for idx, pid in enumerate(candidate_ids)
    }, diagnostics


def _relative_source_rerank(
    candidate_pool: List[Tuple[int, float]],
    stage1_per_sample: List[Dict],
    source_score_map: Dict[int, float],
    top_k_pods: int,
    stage1_weight: float,
    stage2_weight: float,
    source_weight: float,
    keep_top1_anchor: bool,
    anchor_margin: float,
) -> Tuple[
    List[Tuple[int, float]],
    List[Dict],
    List[float],
    List[float],
    List[float],
    List[int],
]:
    if not candidate_pool:
        return [], [], [], [], [], []
    shared_candidate_pool, shared_stage1_vals, shared_stage2_vals, vote_counts = (
        _rerank_shared_with_stage1(
            final_shared_ranking=candidate_pool,
            stage1_per_sample=stage1_per_sample,
            stage1_weight=stage1_weight,
            stage2_weight=stage2_weight,
            keep_top1_anchor=False,
            anchor_margin=anchor_margin,
        )
    )
    pid_list = [int(pid) for pid, _ in shared_candidate_pool]
    shared_source_vals = [float(source_score_map.get(pid, 0.0)) for pid in pid_list]
    stage1_norm = _safe_minmax_norm(th.tensor(shared_stage1_vals, dtype=th.float32))
    stage2_norm = _safe_minmax_norm(th.tensor(shared_stage2_vals, dtype=th.float32))
    source_norm = _safe_minmax_norm(th.tensor(shared_source_vals, dtype=th.float32))
    combined = _combine_score_tensors(
        [
            ("stage1", stage1_norm, stage1_weight),
            ("stage2", stage2_norm, stage2_weight),
            ("relative_source", source_norm, source_weight),
        ],
        device="cpu",
    )
    shared_scores = (
        [float(x) for x in combined.tolist()]
        if combined is not None
        else [0.0 for _ in pid_list]
    )
    if keep_top1_anchor and shared_scores:
        shared_scores[0] = max(shared_scores) + anchor_margin
    final_shared = sorted(
        zip(pid_list, shared_scores), key=lambda item: item[1], reverse=True
    )
    per_sample_results: List[Dict] = []
    for rec in stage1_per_sample or []:
        rec_scores = rec.get("scores") or []
        sample_stage1_vals: List[float] = []
        for pid in pid_list:
            if isinstance(rec_scores, list) and 0 <= pid < len(rec_scores):
                try:
                    sample_stage1_vals.append(float(rec_scores[pid]))
                except Exception:
                    sample_stage1_vals.append(0.0)
            else:
                sample_stage1_vals.append(0.0)
        sample_stage1_norm = _safe_minmax_norm(
            th.tensor(sample_stage1_vals, dtype=th.float32)
        )
        sample_combined = _combine_score_tensors(
            [
                ("stage1", sample_stage1_norm, stage1_weight),
                ("stage2", stage2_norm, stage2_weight),
                ("relative_source", source_norm, source_weight),
            ],
            device="cpu",
        )
        sample_scores = (
            [float(x) for x in sample_combined.tolist()]
            if sample_combined is not None
            else [0.0 for _ in pid_list]
        )
        ranked = sorted(
            zip(pid_list, sample_scores), key=lambda item: item[1], reverse=True
        )
        per_sample_results.append({"final_ranking": ranked[:top_k_pods]})
    return (
        final_shared,
        per_sample_results,
        shared_stage1_vals,
        shared_stage2_vals,
        shared_source_vals,
        vote_counts,
    )


def _failure_type_rerank_profile(failure_type: str) -> Dict[str, float]:
    ft = str(failure_type or "").strip().lower()
    if "replace-body" in ft:
        return {
            "stage1": 0.40,
            "stage2": 0.38,
            "source": 0.22,
            "keep_anchor": 1.0,
            "shared_penalty": 0.48,
        }
    if any(
        key in ft
        for key in (
            "replace-code",
            "replace-method",
            "replace-path",
            "exception",
            "return",
        )
    ):
        return {
            "stage1": 0.20,
            "stage2": 0.55,
            "source": 0.25,
            "keep_anchor": 1.0,
            "shared_penalty": 0.65,
        }
    if any(
        key in ft
        for key in ("stress", "request-abort", "pod-failure", "container-kill")
    ):
        return {
            "stage1": 0.55,
            "stage2": 0.25,
            "source": 0.20,
            "keep_anchor": 0.0,
            "shared_penalty": 1.65,
        }
    if any(key in ft for key in ("response-abort",)):
        return {
            "stage1": 0.30,
            "stage2": 0.45,
            "source": 0.25,
            "keep_anchor": 1.0,
            "shared_penalty": 1.15,
        }
    if any(key in ft for key in ("request-delay", "response-delay")):
        return {
            "stage1": 0.30,
            "stage2": 0.45,
            "source": 0.25,
            "keep_anchor": 1.0,
            "shared_penalty": 0.20,
        }
    if any(
        key in ft
        for key in (
            "partition",
            "bandwidth",
            "loss",
            "mysql",
            "corrupt",
            "time",
            "unknown",
        )
    ):
        return {
            "stage1": 0.60,
            "stage2": 0.20,
            "source": 0.20,
            "keep_anchor": 0.0,
            "shared_penalty": 0.75,
        }
    return {
        "stage1": 0.40,
        "stage2": 0.35,
        "source": 0.25,
        "keep_anchor": 1.0,
        "shared_penalty": 0.85,
    }


def _is_code_family_failure(failure_type: str) -> bool:
    ft = str(failure_type or "").strip().lower()
    return any(key in ft for key in ("replace-code", "replace-method", "replace-body"))


def _apply_code_family_top1_challenger(
    failure_type: str,
    pool: List[int],
    combined_scores: List[float],
    stage1_vals: List[float],
    stage2_vals: List[float],
    source_vals: List[float],
    margin: float,
    topn: int,
    anchor_margin: float,
) -> List[float]:
    if not _is_code_family_failure(failure_type):
        return combined_scores
    if len(pool) < 2 or len(combined_scores) != len(pool):
        return combined_scores
    ranked_indices = sorted(
        range(len(pool)), key=lambda idx: combined_scores[idx], reverse=True
    )
    shortlist = ranked_indices[: max(2, min(int(topn), len(ranked_indices)))]
    incumbent_idx = shortlist[0]
    stage1_norm = _safe_minmax_norm(th.tensor(stage1_vals, dtype=th.float32))
    stage2_norm = _safe_minmax_norm(th.tensor(stage2_vals, dtype=th.float32))
    source_norm = _safe_minmax_norm(th.tensor(source_vals, dtype=th.float32))

    def _norm_at(norm_tensor, idx: int) -> float:
        if norm_tensor is None:
            return 0.0
        try:
            return float(norm_tensor[idx].item())
        except Exception:
            return 0.0

    incumbent_challenger = (
        0.15 * _norm_at(stage1_norm, incumbent_idx)
        + 0.60 * _norm_at(stage2_norm, incumbent_idx)
        + 0.25 * _norm_at(source_norm, incumbent_idx)
    )
    best_idx = incumbent_idx
    best_score = incumbent_challenger
    for idx in shortlist[1:]:
        challenger_score = (
            0.15 * _norm_at(stage1_norm, idx)
            + 0.60 * _norm_at(stage2_norm, idx)
            + 0.25 * _norm_at(source_norm, idx)
        )
        if challenger_score > best_score:
            best_idx = idx
            best_score = challenger_score
    if best_idx == incumbent_idx:
        return combined_scores
    incumbent_stage2 = _norm_at(stage2_norm, incumbent_idx)
    incumbent_source = _norm_at(source_norm, incumbent_idx)
    best_stage2 = _norm_at(stage2_norm, best_idx)
    best_source = _norm_at(source_norm, best_idx)
    has_stronger_signal = best_stage2 >= incumbent_stage2 and best_source >= max(
        incumbent_source - 0.02, 0.0
    )
    if not has_stronger_signal or best_score < incumbent_challenger + float(margin):
        return combined_scores
    adjusted_scores = list(combined_scores)
    adjusted_scores[best_idx] = max(adjusted_scores) + max(anchor_margin, float(margin))
    return adjusted_scores


def _is_top1_trap_failure(failure_type: str) -> bool:
    ft = str(failure_type or "").strip().lower()
    return any(
        key in ft
        for key in (
            "stress",
            "request-abort",
            "pod-failure",
            "response-abort",
            "container-kill",
        )
    )


def _apply_legacy_trap_top1_challenger(
    failure_type: str,
    pool: List[int],
    combined_scores: List[float],
    stage1_vals: List[float],
    stage2_vals: List[float],
    source_vals: List[float],
    anchor_margin: float,
) -> List[float]:
    if not _is_top1_trap_failure(failure_type):
        return combined_scores
    if len(pool) < 2 or len(combined_scores) != len(pool):
        return combined_scores
    ranked_indices = sorted(
        range(len(pool)), key=lambda idx: combined_scores[idx], reverse=True
    )
    incumbent_idx = ranked_indices[0]
    if int(pool[incumbent_idx]) != 22:
        return combined_scores
    shortlist = ranked_indices[: max(2, min(4, len(ranked_indices)))]
    stage1_norm = _safe_minmax_norm(th.tensor(stage1_vals, dtype=th.float32))
    stage2_norm = _safe_minmax_norm(th.tensor(stage2_vals, dtype=th.float32))
    source_norm = _safe_minmax_norm(th.tensor(source_vals, dtype=th.float32))

    def _norm_at(norm_tensor, idx: int) -> float:
        if norm_tensor is None:
            return 0.0
        try:
            return float(norm_tensor[idx].item())
        except Exception:
            return 0.0

    incumbent_s1 = _norm_at(stage1_norm, incumbent_idx)
    incumbent_s2 = _norm_at(stage2_norm, incumbent_idx)
    incumbent_src = _norm_at(source_norm, incumbent_idx)
    best_idx = incumbent_idx
    best_gain = 0.0
    for idx in shortlist[1:]:
        cand_s1 = _norm_at(stage1_norm, idx)
        cand_s2 = _norm_at(stage2_norm, idx)
        cand_src = _norm_at(source_norm, idx)
        gain = (
            0.55 * max(cand_s2 - incumbent_s2, 0.0)
            + 0.30 * max(cand_s1 - incumbent_s1, 0.0)
            + 0.15 * max(cand_src - incumbent_src, 0.0)
        )
        if cand_s2 >= max(incumbent_s2 - 0.03, 0.0) and gain > best_gain:
            best_idx = idx
            best_gain = gain
    if best_idx == incumbent_idx or best_gain < 0.08:
        return combined_scores
    adjusted_scores = list(combined_scores)
    adjusted_scores[best_idx] = max(adjusted_scores) + max(anchor_margin, 1e-4)
    adjusted_scores[incumbent_idx] = min(
        adjusted_scores[incumbent_idx],
        adjusted_scores[best_idx] - max(anchor_margin, 1e-4),
    )
    return adjusted_scores


def _apply_global_hot_candidate_challenger(
    pool: List[int],
    combined_scores: List[float],
    stage1_vals: List[float],
    stage2_vals: List[float],
    source_vals: List[float],
    shared_vals: List[float],
    anchor_margin: float,
    topn: int = 5,
    hot_threshold: float = 0.95,
    source_max: float = 0.05,
    max_final_gap: float = 0.30,
    evidence_margin: float = -0.05,
    hot_debias_weight: float = 0.15,
) -> Tuple[List[float], Dict[str, object]]:
    if len(pool) < 2 or len(combined_scores) != len(pool):
        return combined_scores, {"promoted": False, "reason": "insufficient_candidates"}
    ranked_indices = sorted(
        range(len(pool)), key=lambda idx: combined_scores[idx], reverse=True
    )
    incumbent_idx = ranked_indices[0]
    shortlist = ranked_indices[: max(2, min(int(topn), len(ranked_indices)))]
    stage1_norm = _safe_minmax_norm(th.tensor(stage1_vals, dtype=th.float32))
    stage2_norm = _safe_minmax_norm(th.tensor(stage2_vals, dtype=th.float32))
    source_norm = _safe_minmax_norm(th.tensor(source_vals, dtype=th.float32))

    def _norm_at(norm_tensor, idx: int) -> float:
        if norm_tensor is None:
            return 0.0
        try:
            return float(norm_tensor[idx].item())
        except Exception:
            return 0.0

    incumbent_hot = (
        float(shared_vals[incumbent_idx]) if incumbent_idx < len(shared_vals) else 0.0
    )
    incumbent_src = _norm_at(source_norm, incumbent_idx)
    if incumbent_hot < float(hot_threshold):
        return combined_scores, {
            "promoted": False,
            "reason": "incumbent_not_globally_hot",
            "incumbent_id": int(pool[incumbent_idx]),
            "incumbent_hot_score": incumbent_hot,
        }
    if incumbent_src > float(source_max):
        return combined_scores, {
            "promoted": False,
            "reason": "incumbent_has_source_support",
            "incumbent_id": int(pool[incumbent_idx]),
            "incumbent_source_norm": incumbent_src,
        }
    incumbent_s1 = _norm_at(stage1_norm, incumbent_idx)
    incumbent_s2 = _norm_at(stage2_norm, incumbent_idx)
    incumbent_local = 0.20 * incumbent_s1 + 0.55 * incumbent_s2 + 0.25 * incumbent_src
    incumbent_debiased = incumbent_local - float(hot_debias_weight) * incumbent_hot
    best_idx = incumbent_idx
    best_debiased = incumbent_debiased
    best_local = incumbent_local
    for idx in shortlist[1:]:
        cand_s1 = _norm_at(stage1_norm, idx)
        cand_s2 = _norm_at(stage2_norm, idx)
        cand_src = _norm_at(source_norm, idx)
        cand_hot = float(shared_vals[idx]) if idx < len(shared_vals) else 0.0
        if cand_hot >= incumbent_hot - 0.02:
            continue
        final_gap = float(combined_scores[incumbent_idx]) - float(combined_scores[idx])
        if final_gap > float(max_final_gap):
            continue
        local_evidence = 0.20 * cand_s1 + 0.55 * cand_s2 + 0.25 * cand_src
        if cand_s2 < max(incumbent_s2 - 0.15, 0.0) and cand_src <= incumbent_src + 0.05:
            continue
        debiased_evidence = local_evidence - float(hot_debias_weight) * cand_hot
        if debiased_evidence > best_debiased + float(evidence_margin):
            best_idx = idx
            best_debiased = debiased_evidence
            best_local = local_evidence
    if best_idx == incumbent_idx:
        return combined_scores, {
            "promoted": False,
            "reason": "no_local_challenger",
            "incumbent_id": int(pool[incumbent_idx]),
            "incumbent_hot_score": incumbent_hot,
            "incumbent_local_evidence": float(incumbent_local),
            "incumbent_debiased_evidence": float(incumbent_debiased),
        }
    adjusted_scores = list(combined_scores)
    adjusted_scores[best_idx] = max(adjusted_scores) + max(anchor_margin, 1e-4)
    adjusted_scores[incumbent_idx] = min(
        adjusted_scores[incumbent_idx],
        adjusted_scores[best_idx] - max(anchor_margin, 1e-4),
    )
    return adjusted_scores, {
        "promoted": True,
        "incumbent_id": int(pool[incumbent_idx]),
        "challenger_id": int(pool[best_idx]),
        "incumbent_hot_score": incumbent_hot,
        "challenger_hot_score": (
            float(shared_vals[best_idx]) if best_idx < len(shared_vals) else 0.0
        ),
        "incumbent_local_evidence": float(incumbent_local),
        "challenger_local_evidence": float(best_local),
        "incumbent_debiased_evidence": float(incumbent_debiased),
        "challenger_debiased_evidence": float(best_debiased),
        "final_gap_before": float(combined_scores[incumbent_idx])
        - float(combined_scores[best_idx]),
    }


def _rerank_blend_for_failure_type(
    failure_type: str,
    default_blend: float,
    replace_body_blend: Optional[float] = None,
) -> float:
    ft = str(failure_type or "").strip().lower()
    if "replace-body" in ft:
        rb = 0.35 if replace_body_blend is None else float(replace_body_blend)
        return float(max(0.0, min(1.0, rb)))
    return float(max(0.0, min(1.0, float(default_blend))))


def _apply_stage1_strong_evidence_guard(
    pool: List[int],
    combined_scores: List[float],
    stage1_norm_vals: List[float],
    topn: int,
    bonus: float,
    min_norm: float,
    gap: float,
    preserve_top1: bool,
    anchor_margin: float,
) -> Tuple[List[float], Dict[str, object]]:
    debug: Dict[str, object] = {
        "enabled": True,
        "applied": False,
        "protected_pods": [],
        "reason": "",
    }
    if not pool or not combined_scores or not stage1_norm_vals:
        debug["reason"] = "empty_input"
        return combined_scores, debug
    n = min(len(pool), len(combined_scores), len(stage1_norm_vals))
    if n <= 1:
        debug["reason"] = "too_few_candidates"
        return combined_scores, debug
    topn = max(1, min(int(topn), n))
    bonus = float(max(0.0, bonus))
    min_norm = float(np.clip(min_norm, 0.0, 1.0))
    gap = float(max(0.0, gap))
    if bonus <= 0.0:
        debug["reason"] = "zero_bonus"
        return combined_scores, debug
    ranked_by_stage1 = sorted(
        range(n), key=lambda i: (-float(stage1_norm_vals[i]), int(pool[i]))
    )
    top_score = float(stage1_norm_vals[ranked_by_stage1[0]])
    boundary_idx = ranked_by_stage1[topn] if topn < n else ranked_by_stage1[-1]
    boundary_score = float(stage1_norm_vals[boundary_idx])
    if top_score < min_norm:
        debug["reason"] = "weak_stage1_top"
        debug["top_stage1_norm"] = top_score
        return combined_scores, debug
    if topn < n and (top_score - boundary_score) < gap:
        debug["reason"] = "insufficient_stage1_gap"
        debug["top_stage1_norm"] = top_score
        debug["boundary_stage1_norm"] = boundary_score
        return combined_scores, debug
    updated = list(combined_scores)
    incumbent_idx = max(range(n), key=lambda i: float(combined_scores[i]))
    incumbent_score = float(combined_scores[incumbent_idx])
    cap_score = incumbent_score - max(float(anchor_margin), 1e-6)
    protected: List[Dict[str, object]] = []
    for rank, idx in enumerate(ranked_by_stage1[:topn], start=1):
        s1 = float(stage1_norm_vals[idx])
        if s1 < min_norm:
            continue
        rank_decay = float(topn - rank + 1) / float(topn)
        delta = bonus * rank_decay * s1
        new_score = updated[idx] + delta
        if preserve_top1 and idx != incumbent_idx:
            new_score = min(new_score, cap_score)
        applied_delta = new_score - updated[idx]
        updated[idx] = new_score
        protected.append(
            {
                "pod_id": int(pool[idx]),
                "rank": int(rank),
                "stage1_norm": s1,
                "bonus": float(applied_delta),
            }
        )
    if not protected:
        debug["reason"] = "no_candidate_above_min_norm"
        return combined_scores, debug
    debug["applied"] = True
    debug["protected_pods"] = protected
    debug["top_stage1_norm"] = top_score
    debug["boundary_stage1_norm"] = boundary_score
    debug["preserve_top1"] = bool(preserve_top1)
    debug["incumbent_id"] = int(pool[incumbent_idx])
    return updated, debug


def _confidence_from_norm_values(
    values: Optional[th.Tensor], gap_topn: int
) -> Dict[str, float]:
    if values is None:
        return {"confidence": 0.0, "top1": 0.0, "gap": 0.0, "entropy": 1.0}
    vals = [float(x) for x in values.tolist()]
    if not vals:
        return {"confidence": 0.0, "top1": 0.0, "gap": 0.0, "entropy": 1.0}
    vals_sorted = sorted(vals, reverse=True)
    top1 = vals_sorted[0]
    gap_idx = max(1, min(int(gap_topn), len(vals_sorted))) - 1
    gap = max(0.0, top1 - vals_sorted[gap_idx])
    arr = np.asarray([max(v, 0.0) for v in vals], dtype=np.float64)
    total = float(arr.sum())
    if total <= 1e-12 or len(arr) <= 1:
        entropy_norm = 1.0
    else:
        probs = arr / total
        entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        entropy_norm = float(np.clip(entropy / np.log(len(arr)), 0.0, 1.0))
    sharpness = 1.0 - entropy_norm
    confidence = float(np.clip(0.55 * gap + 0.30 * sharpness + 0.15 * top1, 0.0, 1.0))
    return {
        "confidence": confidence,
        "top1": float(top1),
        "gap": float(gap),
        "entropy": float(entropy_norm),
    }


def _apply_dynamic_confidence_weights(
    stage1_weight: float,
    stage2_weight: float,
    source_weight: float,
    stage1_norm: Optional[th.Tensor],
    stage2_norm: Optional[th.Tensor],
    source_norm: Optional[th.Tensor],
    strength: float,
    gap_topn: int,
    max_shift: float,
) -> Tuple[float, float, float, Dict[str, object]]:
    base_weights = np.asarray(
        [
            max(float(stage1_weight), 0.0),
            max(float(stage2_weight), 0.0),
            max(float(source_weight), 0.0),
        ],
        dtype=np.float64,
    )
    total = float(base_weights.sum())
    if total <= 1e-12:
        return (
            stage1_weight,
            stage2_weight,
            source_weight,
            {"enabled": True, "applied": False, "reason": "zero_base_weight"},
        )
    confs = [
        _confidence_from_norm_values(stage1_norm, gap_topn),
        _confidence_from_norm_values(stage2_norm, gap_topn),
        _confidence_from_norm_values(source_norm, gap_topn),
    ]
    conf_arr = np.asarray([c["confidence"] for c in confs], dtype=np.float64)
    mean_conf = float(conf_arr.mean())
    centered = conf_arr - mean_conf
    shift = np.clip(
        centered * float(max(strength, 0.0)), -float(max_shift), float(max_shift)
    )
    adjusted = np.maximum(base_weights + shift, 0.0)
    adjusted_total = float(adjusted.sum())
    if adjusted_total <= 1e-12:
        adjusted = base_weights.copy()
        adjusted_total = total
    adjusted = adjusted * (total / adjusted_total)
    debug: Dict[str, object] = {
        "enabled": True,
        "applied": True,
        "base_weights": {
            "stage1": float(base_weights[0]),
            "stage2": float(base_weights[1]),
            "source": float(base_weights[2]),
        },
        "adjusted_weights": {
            "stage1": float(adjusted[0]),
            "stage2": float(adjusted[1]),
            "source": float(adjusted[2]),
        },
        "confidence": {
            "stage1": confs[0],
            "stage2": confs[1],
            "source": confs[2],
        },
    }
    return float(adjusted[0]), float(adjusted[1]), float(adjusted[2]), debug


def _failure_type_aware_rerank(
    per_sample_final: List[Dict],
    stage1_per_sample: List[Dict],
    labels: List[Dict],
    top_k_pods: int,
    shared_stage2_scores: List[Tuple[int, float]],
    source_score_map: Dict[int, float],
    default_stage1_weight: float,
    default_stage2_weight: float,
    default_source_weight: float,
    default_shared_penalty: float,
    keep_top1_anchor: bool,
    anchor_margin: float,
    code_top1_challenger_enabled: bool = False,
    code_top1_challenger_topn: int = 4,
    code_top1_challenger_margin: float = 0.02,
    global_hot_challenger_enabled: bool = False,
    global_hot_challenger_topn: int = 5,
    global_hot_challenger_threshold: float = 0.95,
    global_hot_challenger_margin: float = 0.30,
    global_hot_challenger_evidence_margin: float = -0.05,
    stage1_guard_enabled: bool = False,
    stage1_guard_topn: int = 3,
    stage1_guard_bonus: float = 0.06,
    stage1_guard_min_norm: float = 0.75,
    stage1_guard_gap: float = 0.08,
    stage1_guard_preserve_top1: bool = True,
    dynamic_confidence_fusion_enabled: bool = False,
    dynamic_confidence_strength: float = 0.35,
    dynamic_confidence_gap_topn: int = 3,
    dynamic_confidence_max_shift: float = 0.18,
    lgbm_rerank_base_stage2: Optional[Dict[int, float]] = None,
    lgbm_rerank_pred_scaled: Optional[
        Union[Dict[int, float], List[Dict[int, float]]]
    ] = None,
    lgbm_rerank_default_blend: float = 0.75,
    lgbm_rerank_replace_body_blend: Optional[float] = None,
) -> Tuple[List[Dict], List[Tuple[int, float]]]:
    if not per_sample_final or not stage1_per_sample:
        return per_sample_final, []
    stage2_score_map = {int(pid): float(score) for pid, score in shared_stage2_scores}
    shared_candidate_counts: Dict[int, int] = defaultdict(int)
    for rec in stage1_per_sample:
        sample_candidates = rec.get("candidate_services") or []
        seen_local = set()
        for pid in sample_candidates:
            try:
                pid_i = int(pid)
            except Exception:
                continue
            if pid_i in seen_local:
                continue
            seen_local.add(pid_i)
            shared_candidate_counts[pid_i] += 1
    max_shared_count = (
        max(shared_candidate_counts.values()) if shared_candidate_counts else 0
    )
    shared_accumulator: Dict[int, List[float]] = defaultdict(list)
    reranked_results: List[Dict] = []
    use_lgbm_per_type_blend = (
        lgbm_rerank_base_stage2 is not None
        and lgbm_rerank_pred_scaled is not None
        and len(lgbm_rerank_pred_scaled) > 0
    )
    for idx, rec in enumerate(stage1_per_sample):
        label = labels[idx] if idx < len(labels) else {}
        failure_type = label.get("failure_type", "")
        profile = _failure_type_rerank_profile(failure_type)
        stage1_weight = profile.get("stage1", default_stage1_weight)
        stage2_weight = profile.get("stage2", default_stage2_weight)
        source_weight = profile.get("source", default_source_weight)
        shared_penalty_weight = float(profile.get("shared_penalty", 1.0)) * float(
            default_shared_penalty
        )
        local_keep_anchor = bool(profile.get("keep_anchor", 1.0)) and keep_top1_anchor
        base_ranking = (
            per_sample_final[idx].get("final_ranking", [])
            if idx < len(per_sample_final)
            else []
        )
        stage1_scores = rec.get("scores") or []
        sample_candidates = rec.get("candidate_services") or []
        pool: List[int] = []
        seen = set()
        for pid, _ in base_ranking:
            pid_i = int(pid)
            if pid_i not in seen:
                pool.append(pid_i)
                seen.add(pid_i)
        for pid in sample_candidates:
            try:
                pid_i = int(pid)
            except Exception:
                continue
            if pid_i not in seen:
                pool.append(pid_i)
                seen.add(pid_i)
        for pid, _ in shared_stage2_scores[: max(top_k_pods, 8)]:
            pid_i = int(pid)
            if pid_i not in seen:
                pool.append(pid_i)
                seen.add(pid_i)
        if not pool:
            reranked_results.append(
                {"final_ranking": [], "debug_candidates": [], "debug_weights": {}}
            )
            continue
        lgbm_pred_for_sample: Optional[Dict[int, float]] = None
        if use_lgbm_per_type_blend and lgbm_rerank_pred_scaled is not None:
            if isinstance(lgbm_rerank_pred_scaled, list):
                if idx < len(lgbm_rerank_pred_scaled) and isinstance(
                    lgbm_rerank_pred_scaled[idx], dict
                ):
                    lgbm_pred_for_sample = lgbm_rerank_pred_scaled[idx]
            elif isinstance(lgbm_rerank_pred_scaled, dict):
                lgbm_pred_for_sample = lgbm_rerank_pred_scaled
        lgbm_b = (
            _rerank_blend_for_failure_type(
                failure_type, lgbm_rerank_default_blend, lgbm_rerank_replace_body_blend
            )
            if lgbm_pred_for_sample
            else None
        )
        stage1_vals = []
        stage2_vals = []
        source_vals = []
        shared_vals = []
        for pid in pool:
            stage1_vals.append(
                float(stage1_scores[pid])
                if isinstance(stage1_scores, list) and 0 <= pid < len(stage1_scores)
                else 0.0
            )
            pid_i = int(pid)
            if lgbm_pred_for_sample is not None and pid_i in lgbm_pred_for_sample:
                bb = float(
                    (lgbm_rerank_base_stage2 or {}).get(
                        pid_i, stage2_score_map.get(pid_i, 0.0)
                    )
                )
                ps = float(lgbm_pred_for_sample[pid_i])
                bmix = float(lgbm_b) if lgbm_b is not None else 0.0
                stage2_vals.append((1.0 - bmix) * bb + bmix * ps)
            else:
                stage2_vals.append(float(stage2_score_map.get(pid_i, 0.0)))
            source_vals.append(float(source_score_map.get(pid, 0.0)))
            if max_shared_count > 0:
                shared_vals.append(
                    float(shared_candidate_counts.get(pid, 0)) / float(max_shared_count)
                )
            else:
                shared_vals.append(0.0)
        stage1_norm = _safe_minmax_norm(th.tensor(stage1_vals, dtype=th.float32))
        stage2_norm = _safe_minmax_norm(th.tensor(stage2_vals, dtype=th.float32))
        source_norm = _safe_minmax_norm(th.tensor(source_vals, dtype=th.float32))
        shared_norm = _safe_minmax_norm(th.tensor(shared_vals, dtype=th.float32))
        dynamic_conf_debug: Dict[str, object] = {"enabled": False}
        if dynamic_confidence_fusion_enabled:
            stage1_weight, stage2_weight, source_weight, dynamic_conf_debug = (
                _apply_dynamic_confidence_weights(
                    stage1_weight=stage1_weight,
                    stage2_weight=stage2_weight,
                    source_weight=source_weight,
                    stage1_norm=stage1_norm,
                    stage2_norm=stage2_norm,
                    source_norm=source_norm,
                    strength=dynamic_confidence_strength,
                    gap_topn=dynamic_confidence_gap_topn,
                    max_shift=dynamic_confidence_max_shift,
                )
            )
        combined = _combine_score_tensors(
            [
                ("stage1", stage1_norm, stage1_weight),
                ("stage2", stage2_norm, stage2_weight),
                ("relative_source", source_norm, source_weight),
            ],
            device="cpu",
        )
        if combined is None:
            final_ranking = [
                (pid, stage2_score_map.get(pid, 0.0)) for pid in pool[:top_k_pods]
            ]
            dw_fb = {
                "stage1": float(stage1_weight),
                "stage2": float(stage2_weight),
                "source": float(source_weight),
                "shared_penalty": float(shared_penalty_weight),
                "keep_anchor": bool(local_keep_anchor),
                "rerank_mode": "fallback_stage2_only",
            }
            if lgbm_b is not None:
                dw_fb["lgbm_rerank_blend"] = float(lgbm_b)
            reranked_results.append(
                {
                    "final_ranking": final_ranking,
                    "debug_candidates": [
                        {
                            "pod_id": int(pid),
                            "stage1_score": (
                                float(stage1_vals[pos])
                                if pos < len(stage1_vals)
                                else 0.0
                            ),
                            "stage2_score": (
                                float(stage2_vals[pos])
                                if pos < len(stage2_vals)
                                else 0.0
                            ),
                            "source_score": (
                                float(source_vals[pos])
                                if pos < len(source_vals)
                                else 0.0
                            ),
                            "final_score": float(stage2_score_map.get(pid, 0.0)),
                            "in_final_topk": pos < top_k_pods,
                        }
                        for pos, pid in enumerate(pool)
                    ],
                    "debug_weights": dw_fb,
                }
            )
            for pid, score in final_ranking:
                shared_accumulator[int(pid)].append(float(score))
            continue
        combined_scores = [float(x) for x in combined.tolist()]
        if shared_norm is not None and shared_penalty_weight > 0:
            shared_penalties = [
                float(x) * float(shared_penalty_weight) for x in shared_norm.tolist()
            ]
            combined_scores = [
                score - penalty
                for score, penalty in zip(combined_scores, shared_penalties)
            ]
        if combined_scores and local_keep_anchor:
            combined_scores[0] = max(combined_scores) + anchor_margin
        stage1_guard_debug: Dict[str, object] = {"enabled": False}
        if stage1_guard_enabled:
            stage1_norm_vals = (
                [float(x) for x in stage1_norm.tolist()]
                if stage1_norm is not None
                else []
            )
            combined_scores, stage1_guard_debug = _apply_stage1_strong_evidence_guard(
                pool=pool,
                combined_scores=combined_scores,
                stage1_norm_vals=stage1_norm_vals,
                topn=stage1_guard_topn,
                bonus=stage1_guard_bonus,
                min_norm=stage1_guard_min_norm,
                gap=stage1_guard_gap,
                preserve_top1=stage1_guard_preserve_top1,
                anchor_margin=anchor_margin,
            )
        if code_top1_challenger_enabled:
            combined_scores = _apply_code_family_top1_challenger(
                failure_type=failure_type,
                pool=pool,
                combined_scores=combined_scores,
                stage1_vals=stage1_vals,
                stage2_vals=stage2_vals,
                source_vals=source_vals,
                margin=code_top1_challenger_margin,
                topn=code_top1_challenger_topn,
                anchor_margin=anchor_margin,
            )
        combined_scores = _apply_legacy_trap_top1_challenger(
            failure_type=failure_type,
            pool=pool,
            combined_scores=combined_scores,
            stage1_vals=stage1_vals,
            stage2_vals=stage2_vals,
            source_vals=source_vals,
            anchor_margin=anchor_margin,
        )
        hot_challenger_debug: Dict[str, object] = {
            "promoted": False,
            "reason": "disabled",
        }
        if global_hot_challenger_enabled:
            combined_scores, hot_challenger_debug = (
                _apply_global_hot_candidate_challenger(
                    pool=pool,
                    combined_scores=combined_scores,
                    stage1_vals=stage1_vals,
                    stage2_vals=stage2_vals,
                    source_vals=source_vals,
                    shared_vals=shared_vals,
                    anchor_margin=anchor_margin,
                    topn=global_hot_challenger_topn,
                    hot_threshold=global_hot_challenger_threshold,
                    max_final_gap=global_hot_challenger_margin,
                    evidence_margin=global_hot_challenger_evidence_margin,
                )
            )
        debug_candidates = sorted(
            [
                {
                    "pod_id": int(pid),
                    "stage1_score": (
                        float(stage1_vals[pos]) if pos < len(stage1_vals) else 0.0
                    ),
                    "stage2_score": (
                        float(stage2_vals[pos]) if pos < len(stage2_vals) else 0.0
                    ),
                    "source_score": (
                        float(source_vals[pos]) if pos < len(source_vals) else 0.0
                    ),
                    "shared_candidate_score": (
                        float(shared_vals[pos]) if pos < len(shared_vals) else 0.0
                    ),
                    "final_score": (
                        float(combined_scores[pos])
                        if pos < len(combined_scores)
                        else 0.0
                    ),
                }
                for pos, pid in enumerate(pool)
            ],
            key=lambda item: item["final_score"],
            reverse=True,
        )
        final_ranking = [
            (item["pod_id"], item["final_score"])
            for item in debug_candidates[:top_k_pods]
        ]
        dw_ok = {
            "stage1": float(stage1_weight),
            "stage2": float(stage2_weight),
            "source": float(source_weight),
            "shared_penalty": float(shared_penalty_weight),
            "keep_anchor": bool(local_keep_anchor),
            "rerank_mode": "type_aware",
            "global_hot_challenger": hot_challenger_debug,
            "stage1_guard": stage1_guard_debug,
            "dynamic_confidence_fusion": dynamic_conf_debug,
        }
        if lgbm_b is not None:
            dw_ok["lgbm_rerank_blend"] = float(lgbm_b)
        reranked_results.append(
            {
                "final_ranking": final_ranking,
                "debug_candidates": debug_candidates,
                "debug_weights": dw_ok,
            }
        )
        for pid, score in final_ranking:
            shared_accumulator[int(pid)].append(float(score))
    shared_ranking = sorted(
        [
            (pid, float(np.mean(scores)))
            for pid, scores in shared_accumulator.items()
            if scores
        ],
        key=lambda item: item[1],
        reverse=True,
    )[:top_k_pods]
    return reranked_results, shared_ranking


def _leave_one_out_stage2_rerank(
    candidate_pool: List[Tuple[int, float]],
    stage1_per_sample: List[Dict],
    groundtruths: List[Set[Tuple[str, int]]],
    top_k_pods: int,
    stage1_weight: float,
    stage2_weight: float,
    blend: float,
    epochs: int,
    lr: float,
    l2: float,
) -> Tuple[List[Dict], List[Tuple[int, float]], Dict[str, object]]:
    if not candidate_pool or not stage1_per_sample or not groundtruths:
        return [], [], {"enabled": False, "reason": "empty_input"}
    candidate_ids = [int(pid) for pid, _ in candidate_pool]
    stage2_raw = [float(score) for _, score in candidate_pool]
    stage2_norm_tensor = _safe_minmax_norm(th.tensor(stage2_raw, dtype=th.float32))
    stage2_norm = (
        [float(x) for x in stage2_norm_tensor.tolist()]
        if stage2_norm_tensor is not None
        else [0.0] * len(candidate_ids)
    )
    blend = float(np.clip(blend, 0.0, 1.0))
    shortlist_size = max(3, min(6, len(candidate_ids)))
    sample_features: List[Dict[str, object]] = []
    for rec in stage1_per_sample:
        scores = rec.get("scores") or []
        sample_candidate_ids = {
            int(pid)
            for pid in (rec.get("candidate_services") or [])
            if isinstance(pid, (int, np.integer)) or str(pid).isdigit()
        }
        s1_vals = [
            float(scores[pid]) if 0 <= pid < len(scores) else 0.0
            for pid in candidate_ids
        ]
        s1_norm_tensor = _safe_minmax_norm(th.tensor(s1_vals, dtype=th.float32))
        s1_norm = (
            [float(x) for x in s1_norm_tensor.tolist()]
            if s1_norm_tensor is not None
            else [0.0] * len(candidate_ids)
        )
        base_tensor = _combine_score_tensors(
            [
                ("stage1", th.tensor(s1_norm, dtype=th.float32), stage1_weight),
                ("stage2", th.tensor(stage2_norm, dtype=th.float32), stage2_weight),
            ],
            device="cpu",
        )
        base_scores = (
            [float(x) for x in base_tensor.tolist()]
            if base_tensor is not None
            else [0.0] * len(candidate_ids)
        )
        base_norm_tensor = _safe_minmax_norm(th.tensor(base_scores, dtype=th.float32))
        base_norm = (
            [float(x) for x in base_norm_tensor.tolist()]
            if base_norm_tensor is not None
            else [0.0] * len(candidate_ids)
        )
        s1_rank = np.argsort(-np.asarray(s1_vals, dtype=np.float32))
        s1_rank_pos = np.empty(len(candidate_ids), dtype=np.float32)
        for rank, idx in enumerate(s1_rank):
            s1_rank_pos[idx] = rank
        inv_s1_rank = (
            1.0 - (s1_rank_pos / max(len(candidate_ids) - 1, 1))
            if candidate_ids
            else np.zeros(0, dtype=np.float32)
        )
        in_sample = np.asarray(
            [1.0 if pid in sample_candidate_ids else 0.0 for pid in candidate_ids],
            dtype=np.float32,
        )
        s1_arr = np.asarray(s1_norm, dtype=np.float32)
        s2_arr = np.asarray(stage2_norm, dtype=np.float32)
        base_arr = np.asarray(base_norm, dtype=np.float32)
        feat_mat = np.stack(
            [
                base_arr,
                s1_arr,
                s2_arr,
                s1_arr * s2_arr,
                np.maximum(s1_arr - s2_arr, 0.0),
                np.maximum(s2_arr - s1_arr, 0.0),
                in_sample,
                inv_s1_rank,
            ],
            axis=1,
        )
        base_rank = list(np.argsort(-base_arr))
        sample_features.append(
            {
                "features": th.tensor(feat_mat, dtype=th.float32),
                "base_scores": base_scores,
                "base_norm": base_norm,
                "stage1_norm": s1_norm,
                "stage2_norm": stage2_norm,
                "base_rank": base_rank,
            }
        )

    def _target_pods(gt: Set[Tuple[str, int]]) -> Set[int]:
        targets: Set[int] = set()
        for item in gt or set():
            try:
                ntype, nid = item
            except Exception:
                continue
            if str(ntype) != "pod":
                continue
            try:
                targets.add(int(nid))
            except Exception:
                pass
        return targets

    gt_pods = [_target_pods(gt) for gt in groundtruths]
    reranked_per_sample: List[Dict] = []
    shared_accumulator = np.zeros(len(candidate_ids), dtype=np.float64)
    train_losses: List[float] = []
    learned_weight_rows: List[List[float]] = []
    challenger_promotions = 0
    for holdout_idx, feats in enumerate(sample_features):
        train_x_list: List[th.Tensor] = []
        train_y_list: List[th.Tensor] = []
        for train_idx, train_feats in enumerate(sample_features):
            if train_idx == holdout_idx:
                continue
            targets = gt_pods[train_idx] if train_idx < len(gt_pods) else set()
            shortlist = train_feats["base_rank"][:shortlist_size]
            if not shortlist:
                continue
            incumbent_idx = int(shortlist[0])
            positive_shortlist = [
                idx
                for idx in shortlist
                if candidate_ids[idx] in targets and idx != incumbent_idx
            ]
            if not positive_shortlist:
                continue
            incumbent_feat = train_feats["features"][incumbent_idx]
            incumbent_s1 = float(train_feats["stage1_norm"][incumbent_idx])
            incumbent_s2 = float(train_feats["stage2_norm"][incumbent_idx])
            for idx in shortlist[1:]:
                cand_feat = train_feats["features"][idx]
                cand_s1 = float(train_feats["stage1_norm"][idx])
                cand_s2 = float(train_feats["stage2_norm"][idx])
                extra = th.tensor(
                    [
                        cand_s1 - incumbent_s1,
                        cand_s2 - incumbent_s2,
                        max(cand_s1 - incumbent_s1, 0.0),
                        max(cand_s2 - incumbent_s2, 0.0),
                    ],
                    dtype=th.float32,
                )
                pair_feat = th.cat([cand_feat - incumbent_feat, extra], dim=0)
                label = 1.0 if idx in positive_shortlist else 0.0
                train_x_list.append(pair_feat.unsqueeze(0))
                train_y_list.append(th.tensor([label], dtype=th.float32))
        holdout_base = np.asarray(feats["base_norm"], dtype=np.float32)
        if not train_x_list:
            final_scores = holdout_base.tolist()
            ranked_idx = sorted(
                range(len(candidate_ids)),
                key=lambda idx: final_scores[idx],
                reverse=True,
            )
            reranked_per_sample.append(
                {
                    "final_ranking": [
                        (candidate_ids[idx], final_scores[idx])
                        for idx in ranked_idx[:top_k_pods]
                    ]
                }
            )
            shared_accumulator += np.asarray(final_scores, dtype=np.float64)
            continue
        train_x = th.cat(train_x_list, dim=0)
        train_y = th.cat(train_y_list, dim=0)
        model = nn.Linear(train_x.shape[1], 1)
        optimizer = th.optim.AdamW(
            model.parameters(), lr=float(lr), weight_decay=float(l2)
        )
        pos_count = float(train_y.sum().item())
        neg_count = max(float(train_y.numel()) - pos_count, 1.0)
        pos_weight = float(np.clip(neg_count / max(pos_count, 1.0), 1.0, 20.0))
        best_state = deepcopy(model.state_dict())
        best_loss = float("inf")
        num_epochs = max(int(epochs), 20)
        for _ in range(num_epochs):
            optimizer.zero_grad()
            logits = model(train_x).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                train_y,
                pos_weight=th.tensor(pos_weight, dtype=th.float32),
            )
            loss.backward()
            optimizer.step()
            cur_loss = float(loss.item())
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_state = deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        train_losses.append(best_loss)
        learned_weight_rows.append(model.weight.detach().cpu().view(-1).tolist())
        final_scores = holdout_base.copy()
        shortlist = feats["base_rank"][:shortlist_size]
        if shortlist:
            incumbent_idx = int(shortlist[0])
            incumbent_feat = feats["features"][incumbent_idx]
            incumbent_s1 = float(feats["stage1_norm"][incumbent_idx])
            incumbent_s2 = float(feats["stage2_norm"][incumbent_idx])
            best_idx = incumbent_idx
            best_prob = 0.0
            best_local_gain = 0.0
            for idx in shortlist[1:]:
                cand_feat = feats["features"][idx]
                cand_s1 = float(feats["stage1_norm"][idx])
                cand_s2 = float(feats["stage2_norm"][idx])
                extra = th.tensor(
                    [
                        cand_s1 - incumbent_s1,
                        cand_s2 - incumbent_s2,
                        max(cand_s1 - incumbent_s1, 0.0),
                        max(cand_s2 - incumbent_s2, 0.0),
                    ],
                    dtype=th.float32,
                )
                pair_feat = th.cat(
                    [cand_feat - incumbent_feat, extra], dim=0
                ).unsqueeze(0)
                prob = float(th.sigmoid(model(pair_feat)).item())
                local_gain = (
                    0.55 * max(cand_s2 - incumbent_s2, 0.0)
                    + 0.35 * max(cand_s1 - incumbent_s1, 0.0)
                    + 0.10
                    * max(
                        float(feats["base_norm"][idx])
                        - float(feats["base_norm"][incumbent_idx]),
                        0.0,
                    )
                )
                if prob > best_prob:
                    best_prob = prob
                    best_idx = idx
                    best_local_gain = local_gain
            should_promote = (
                best_idx != incumbent_idx
                and best_prob >= 0.58
                and best_local_gain >= 0.03
            )
            if should_promote:
                incumbent_score = float(final_scores[incumbent_idx])
                challenger_score = float(final_scores[best_idx])
                final_scores[best_idx] = max(incumbent_score + 1e-4, challenger_score)
                final_scores[incumbent_idx] = min(
                    incumbent_score, final_scores[best_idx] - 1e-4
                )
                challenger_promotions += 1
        shared_accumulator += np.asarray(final_scores, dtype=np.float64)
        ranked_idx = sorted(
            range(len(candidate_ids)), key=lambda idx: final_scores[idx], reverse=True
        )
        reranked_per_sample.append(
            {
                "final_ranking": [
                    (candidate_ids[idx], final_scores[idx])
                    for idx in ranked_idx[:top_k_pods]
                ]
            }
        )
    shared_scores = (shared_accumulator / max(len(sample_features), 1)).tolist()
    shared_ranked_idx = sorted(
        range(len(candidate_ids)), key=lambda idx: shared_scores[idx], reverse=True
    )
    reranked_shared = [
        (candidate_ids[idx], float(shared_scores[idx]))
        for idx in shared_ranked_idx[:top_k_pods]
    ]
    diag: Dict[str, object] = {
        "enabled": True,
        "mode": "leave_one_out_local_challenger",
        "candidate_pool_size": len(candidate_ids),
        "blend": blend,
        "feature_names": [
            "base_norm",
            "stage1_norm",
            "stage2_norm",
            "stage1_stage2_interaction",
            "stage1_advantage",
            "stage2_advantage",
            "in_sample_candidates",
            "inv_stage1_rank",
        ],
        "shortlist_size": shortlist_size,
        "epochs": max(int(epochs), 20),
        "lr": float(lr),
        "l2": float(l2),
        "challenger_promotions": int(challenger_promotions),
        "avg_train_loss": float(np.mean(train_losses)) if train_losses else None,
        "avg_weights": (
            np.mean(np.asarray(learned_weight_rows, dtype=np.float32), axis=0).tolist()
            if learned_weight_rows
            else []
        ),
    }
    return reranked_per_sample, reranked_shared, diag


def _lag_aware_dual_stage_rca_active(
    graphs: Union[DGLGraph, List[DGLGraph]],
    stacked_nfeat: Dict[str, th.Tensor],
    data_stats: Dict[str, Dict[str, th.Tensor]],
    labels: List[Dict],
    device: str = "cpu",
    tau_max: int = 3,
    top_k_services: int = 10,
    top_k_pods: int = 10,
    service_timeseries: Optional[th.Tensor] = None,
    service_list: Optional[List[str]] = None,
    G_trace_edges: Optional[List[Tuple[str, str]]] = None,
    propagation_lambda: float = 0.6,
    **kwargs,
) -> Dict:
    logger.info("=" * 60)
    logger.info("Lag-aware dual-stage RCA")
    logger.info("=" * 60)
    if isinstance(graphs, list):
        graph_list = graphs
        g = graphs[0] if graphs else None
    else:
        graph_list = [graphs]
        g = graphs
    if g is None:
        logger.error("No graph data available for RCA")
        return {}
    if (
        service_timeseries is not None
        and service_list is not None
        and G_trace_edges is not None
    ):
        logger.info("Running constrained PCMCI+ on service-level time series")
        P_prior = constrained_pcmci_plus_with_timeseries(
            service_timeseries=service_timeseries,
            service_list=service_list,
            G_trace_edges=G_trace_edges,
            tau_max=tau_max,
            alpha=0.05,
            feat_idx=0,
        )
    else:
        logger.info("Running constrained PCMCI+ on stacked node features")
        ntype_for_pcmci = "api"
        for candidate_ntype in ["pod", "api"]:
            if candidate_ntype in stacked_nfeat:
                ntype_for_pcmci = candidate_ntype
                break
        P_prior = constrained_pcmci_plus(
            g,
            stacked_nfeat,
            labels,
            ntype=ntype_for_pcmci,
            feat_idx=0,
            tau_max=tau_max,
            alpha=0.05,
            nan_nodes=kwargs.get("nan_nodes", None),
        )
    rca_config = kwargs.get("rca_config", None)
    stage1_expand_factor = getattr(rca_config, "stage1_expand_factor", 2.0)
    stage1_gnn_weight = getattr(rca_config, "stage1_gnn_weight", 0.5)
    stage1_anomaly_weight = getattr(rca_config, "stage1_anomaly_weight", 0.5)
    stage1_response_weight = getattr(rca_config, "stage1_response_weight", 0.55)
    stage1_vote_weight = getattr(rca_config, "stage1_vote_weight", 0.33)
    stage1_mean_weight = getattr(rca_config, "stage1_mean_weight", 0.33)
    stage1_peak_weight = getattr(rca_config, "stage1_peak_weight", 0.34)
    stage1_dual_hit_bonus = getattr(rca_config, "stage1_dual_hit_bonus", 0.10)
    temporal_response_channel_enabled = bool(
        getattr(rca_config, "temporal_response_channel_enabled", True)
    )
    dual_channel_consistency_enabled = bool(
        getattr(rca_config, "dual_channel_consistency_enabled", True)
    )
    stage2_stage1_weight = getattr(rca_config, "stage2_stage1_weight", 0.35)
    stage2_stage2_weight = getattr(rca_config, "stage2_stage2_weight", 0.65)
    stage2_keep_top1_anchor = getattr(rca_config, "stage2_keep_top1_anchor", False)
    stage2_anchor_margin = getattr(rca_config, "stage2_anchor_margin", 1e-6)
    stage2_score_transform = getattr(rca_config, "stage2_score_transform", "log1p_clip")
    stage2_score_clip_percentile = getattr(
        rca_config, "stage2_score_clip_percentile", 95.0
    )
    stage2_score_clip_min_candidates = getattr(
        rca_config, "stage2_score_clip_min_candidates", 8
    )
    stage2_shared_candidate_penalty = float(
        getattr(rca_config, "stage2_shared_candidate_penalty", 0.12)
    )
    type_aware_rerank_enabled = bool(
        getattr(rca_config, "type_aware_rerank_enabled", True)
    )
    stage1_channel_aux = kwargs.get("stage1_channel_aux")
    stage2_source_bonus_weight = getattr(rca_config, "stage2_source_bonus_weight", 0.15)
    stage2_sink_penalty_weight = getattr(rca_config, "stage2_sink_penalty_weight", 0.50)
    stage2_source_first_weight = getattr(rca_config, "stage2_source_first_weight", 0.45)
    source_candidate_pool = max(
        int(getattr(rca_config, "source_candidate_pool", max(top_k_pods, 15))),
        top_k_pods,
    )
    source_score_weight = getattr(rca_config, "source_score_weight", 0.25)
    enable_stage2_reranker = getattr(rca_config, "enable_stage2_reranker", True)
    reranker_candidate_pool = max(
        int(getattr(rca_config, "reranker_candidate_pool", max(top_k_pods, 24))),
        top_k_pods,
    )
    reranker_blend = getattr(rca_config, "reranker_blend", 0.75)
    reranker_blend_replace_body = getattr(
        rca_config, "reranker_blend_replace_body", None
    )
    reranker_epochs = getattr(rca_config, "reranker_epochs", 200)
    reranker_lr = getattr(rca_config, "reranker_lr", 0.05)
    reranker_l2 = getattr(rca_config, "reranker_l2", 1e-4)
    code_top1_challenger_enabled = getattr(
        rca_config, "code_top1_challenger_enabled", True
    )
    code_top1_challenger_topn = max(
        int(getattr(rca_config, "code_top1_challenger_topn", 4)), 2
    )
    code_top1_challenger_margin = float(
        getattr(rca_config, "code_top1_challenger_margin", 0.02)
    )
    global_hot_challenger_enabled = getattr(
        rca_config, "global_hot_challenger_enabled", False
    )
    global_hot_challenger_topn = max(
        int(getattr(rca_config, "global_hot_challenger_topn", 5)), 2
    )
    global_hot_challenger_threshold = float(
        getattr(rca_config, "global_hot_challenger_threshold", 0.95)
    )
    global_hot_challenger_margin = float(
        getattr(rca_config, "global_hot_challenger_margin", 0.30)
    )
    global_hot_challenger_evidence_margin = float(
        getattr(rca_config, "global_hot_challenger_evidence_margin", -0.05)
    )
    stage1_guard_enabled = bool(getattr(rca_config, "stage1_guard_enabled", False))
    stage1_guard_topn = max(int(getattr(rca_config, "stage1_guard_topn", 3)), 1)
    stage1_guard_bonus = float(getattr(rca_config, "stage1_guard_bonus", 0.06))
    stage1_guard_min_norm = float(getattr(rca_config, "stage1_guard_min_norm", 0.75))
    stage1_guard_gap = float(getattr(rca_config, "stage1_guard_gap", 0.08))
    stage1_guard_preserve_top1 = bool(
        getattr(rca_config, "stage1_guard_preserve_top1", True)
    )
    dynamic_confidence_fusion_enabled = bool(
        getattr(rca_config, "dynamic_confidence_fusion_enabled", False)
    )
    dynamic_confidence_strength = float(
        getattr(rca_config, "dynamic_confidence_strength", 0.35)
    )
    dynamic_confidence_gap_topn = max(
        int(getattr(rca_config, "dynamic_confidence_gap_topn", 3)), 2
    )
    dynamic_confidence_max_shift = float(
        getattr(rca_config, "dynamic_confidence_max_shift", 0.18)
    )
    stage1_source_aware_score_enabled = bool(
        getattr(rca_config, "stage1_source_aware_score_enabled", False)
    )
    stage1_source_aware_strength = float(
        getattr(rca_config, "stage1_source_aware_strength", 1.0)
    )
    lag_prior_enabled = bool(getattr(rca_config, "lag_prior_enabled", True))
    lag_alignment_enabled = bool(getattr(rca_config, "lag_alignment_enabled", True))
    counterfactual_effect_enabled = bool(
        getattr(rca_config, "counterfactual_effect_enabled", True)
    )
    local_anomaly_score_enabled = bool(
        getattr(rca_config, "local_anomaly_score_enabled", True)
    )
    if not lag_prior_enabled:
        logger.info(
            "Stage 1.1 prior ablation enabled: dropping lag-causal prior after skeleton discovery"
        )
        P_prior = {}
    ntype = None
    for candidate_ntype in ["pod", "api"]:
        if candidate_ntype in stacked_nfeat:
            ntype = candidate_ntype
            break
    if ntype is None:
        ntype = list(stacked_nfeat.keys())[0] if stacked_nfeat else "api"
        logger.warning(f"Falling back to node type: {ntype}")
    in_feats = stacked_nfeat[ntype].shape[-1] if ntype in stacked_nfeat else 10
    model = LagAwareGNN(
        in_feats=in_feats,
        hidden_feats=64,
        out_feats=32,
        tau_max=tau_max,
        num_heads=4,
        num_layers=2,
    ).to(device)
    stage1_weak_type_conditional_expansion_enabled = bool(
        getattr(rca_config, "stage1_weak_type_conditional_expansion_enabled", False)
    )
    stage1_weak_type_expand_factor = float(
        getattr(rca_config, "stage1_weak_type_expand_factor", 2.0)
    )
    raw_weak_failure_types = getattr(
        rca_config,
        "stage1_weak_failure_types",
        "stress,container-kill,pod-failure,exception,corrupt",
    )
    if isinstance(raw_weak_failure_types, str):
        stage1_weak_failure_types = [
            ft.strip() for ft in raw_weak_failure_types.split(",") if ft.strip()
        ]
    else:
        stage1_weak_failure_types = list(raw_weak_failure_types or [])
    groundtruths = kwargs.get("groundtruths", []) or []
    logger.info("\n[Stage 1] Fault-type-aware dual-channel recall")
    S_cand, tau_star, stage1_per_sample = stage1_fault_type_dual_channel_localization(
        graph_list,
        stacked_nfeat,
        data_stats,
        labels,
        model,
        tau_max,
        P_prior,
        device,
        top_k_services,
        ntype=ntype,
        expand_factor=stage1_expand_factor,
        gnn_weight=stage1_gnn_weight,
        anomaly_weight=stage1_anomaly_weight,
        response_weight=stage1_response_weight,
        vote_weight=stage1_vote_weight,
        mean_weight=stage1_mean_weight,
        peak_weight=stage1_peak_weight,
        dual_hit_bonus=stage1_dual_hit_bonus,
        stage1_channel_aux=stage1_channel_aux,
        temporal_response_channel_enabled=temporal_response_channel_enabled,
        dual_channel_consistency_enabled=dual_channel_consistency_enabled,
        source_aware_enabled=stage1_source_aware_score_enabled,
        source_aware_strength=stage1_source_aware_strength,
        weak_type_conditional_expansion_enabled=stage1_weak_type_conditional_expansion_enabled,
        weak_type_expand_factor=stage1_weak_type_expand_factor,
        weak_failure_types=stage1_weak_failure_types,
    )
    logger.info("\n[Stage 2] Fine-grained localization")
    pod_feats = kwargs.get("pod_feats", {})
    norm_pod_feats = kwargs.get("norm_pod_feats", {})
    downstream_feats = kwargs.get("downstream_feats", {})
    pod_scores = stage2_fine_grained_localization(
        S_cand,
        tau_star,
        pod_feats,
        norm_pod_feats,
        downstream_feats,
        predictor_model=None,
        lambda_1=getattr(rca_config, "lambda_1", 0.6),
        lambda_2=getattr(rca_config, "lambda_2", 0.4),
        lag_alignment_enabled=lag_alignment_enabled,
        counterfactual_effect_enabled=counterfactual_effect_enabled,
        local_anomaly_score_enabled=local_anomaly_score_enabled,
    )
    pruned_pod_scores = node_confounder_pruning(pod_scores, P_prior)
    transformed_pod_scores, stage2_transform_diag = _transform_stage2_score_pairs(
        pruned_pod_scores,
        transform=stage2_score_transform,
        clip_percentile=stage2_score_clip_percentile,
        clip_min_candidates=stage2_score_clip_min_candidates,
    )
    candidate_pool = transformed_pod_scores[:source_candidate_pool]
    source_score_map, source_diag_map = _compute_relative_source_signal(
        candidate_pool=candidate_pool,
        tau_star=tau_star,
        pod_feats=pod_feats,
        norm_pod_feats=norm_pod_feats,
        downstream_feats=downstream_feats,
        lag_alignment_enabled=lag_alignment_enabled,
    )
    (
        relative_shared_ranking,
        per_sample_final,
        stage1_vals,
        stage2_vals,
        source_vals,
        vote_counts,
    ) = _relative_source_rerank(
        candidate_pool=candidate_pool,
        stage1_per_sample=stage1_per_sample,
        source_score_map=source_score_map,
        top_k_pods=top_k_pods,
        stage1_weight=stage2_stage1_weight,
        stage2_weight=stage2_stage2_weight,
        source_weight=source_score_weight,
        keep_top1_anchor=stage2_keep_top1_anchor,
        anchor_margin=stage2_anchor_margin,
    )
    final_shared_ranking = relative_shared_ranking[:top_k_pods]
    if stage1_per_sample:
        per_sample_final = _build_per_sample_final_rankings_true_stage2(
            stage1_per_sample=stage1_per_sample,
            tau_star=tau_star,
            pod_feats=pod_feats,
            norm_pod_feats=norm_pod_feats,
            downstream_feats=downstream_feats,
            pcmci_results=P_prior,
            top_k_pods=top_k_pods,
            stage1_weight=stage2_stage1_weight,
            stage2_weight=stage2_stage2_weight,
            lambda_1=getattr(rca_config, "lambda_1", 0.6),
            lambda_2=getattr(rca_config, "lambda_2", 0.4),
            lag_alignment_enabled=lag_alignment_enabled,
            counterfactual_effect_enabled=counterfactual_effect_enabled,
            local_anomaly_score_enabled=local_anomaly_score_enabled,
            source_bonus_weight=stage2_source_bonus_weight,
            sink_penalty_weight=stage2_sink_penalty_weight,
            source_first_weight=stage2_source_first_weight,
            shared_stage2_fallback=relative_shared_ranking,
        )
    elif not per_sample_final:
        per_sample_final = _build_per_sample_final_rankings(
            final_shared_ranking=final_shared_ranking,
            stage1_per_sample=stage1_per_sample,
            top_k_pods=top_k_pods,
            stage1_weight=stage2_stage1_weight,
            stage2_weight=stage2_stage2_weight,
            keep_top1_anchor=stage2_keep_top1_anchor,
            anchor_margin=stage2_anchor_margin,
        )
    reranker_diag = {
        "enabled": False,
        "requested": bool(enable_stage2_reranker),
        "candidate_pool_size": 0,
        "stage1_samples": len(stage1_per_sample),
        "groundtruths": len(groundtruths),
    }
    per_sample_pre_type_aware = deepcopy(per_sample_final) if per_sample_final else []
    if type_aware_rerank_enabled:
        type_aware_per_sample, type_aware_shared = _failure_type_aware_rerank(
            per_sample_final=per_sample_final,
            stage1_per_sample=stage1_per_sample,
            labels=labels,
            top_k_pods=top_k_pods,
            shared_stage2_scores=transformed_pod_scores,
            source_score_map=source_score_map,
            default_stage1_weight=stage2_stage1_weight,
            default_stage2_weight=stage2_stage2_weight,
            default_source_weight=source_score_weight,
            default_shared_penalty=stage2_shared_candidate_penalty,
            keep_top1_anchor=stage2_keep_top1_anchor,
            anchor_margin=stage2_anchor_margin,
            code_top1_challenger_enabled=code_top1_challenger_enabled,
            code_top1_challenger_topn=code_top1_challenger_topn,
            code_top1_challenger_margin=code_top1_challenger_margin,
            global_hot_challenger_enabled=global_hot_challenger_enabled,
            global_hot_challenger_topn=global_hot_challenger_topn,
            global_hot_challenger_threshold=global_hot_challenger_threshold,
            global_hot_challenger_margin=global_hot_challenger_margin,
            global_hot_challenger_evidence_margin=global_hot_challenger_evidence_margin,
            stage1_guard_enabled=stage1_guard_enabled,
            stage1_guard_topn=stage1_guard_topn,
            stage1_guard_bonus=stage1_guard_bonus,
            stage1_guard_min_norm=stage1_guard_min_norm,
            stage1_guard_gap=stage1_guard_gap,
            stage1_guard_preserve_top1=stage1_guard_preserve_top1,
            dynamic_confidence_fusion_enabled=dynamic_confidence_fusion_enabled,
            dynamic_confidence_strength=dynamic_confidence_strength,
            dynamic_confidence_gap_topn=dynamic_confidence_gap_topn,
            dynamic_confidence_max_shift=dynamic_confidence_max_shift,
        )
        if type_aware_per_sample:
            per_sample_final = type_aware_per_sample
        if type_aware_shared:
            final_shared_ranking = type_aware_shared[:top_k_pods]
    reranker_dump_path = kwargs.get("stage2_reranker_dump_path")
    reranker_model_path = kwargs.get("stage2_reranker_model_path")

    def _build_union_reranker_pool() -> List[Tuple[int, float]]:
        score_map: Dict[int, float] = {
            int(pid): float(score) for pid, score in transformed_pod_scores
        }
        ordered_ids: List[int] = []
        seen_ids: Set[int] = set()

        def _add_pid(pid_obj) -> None:
            try:
                pid_i = int(pid_obj)
            except Exception:
                return
            if pid_i in seen_ids:
                return
            seen_ids.add(pid_i)
            ordered_ids.append(pid_i)

        for pid, _ in final_shared_ranking[:reranker_candidate_pool]:
            _add_pid(pid)
        for pid, _ in transformed_pod_scores[:reranker_candidate_pool]:
            _add_pid(pid)
        for rec in per_sample_pre_type_aware:
            for item in rec.get("final_ranking", []) if isinstance(rec, dict) else []:
                if isinstance(item, (list, tuple)) and item:
                    _add_pid(item[0])
                elif isinstance(item, dict):
                    _add_pid(item.get("pod_id", item.get("id")))
        for rec in stage1_per_sample:
            for pid in (
                (rec.get("candidate_services") or []) if isinstance(rec, dict) else []
            ):
                _add_pid(pid)
        return [(pid, float(score_map.get(pid, 0.0))) for pid in ordered_ids]

    def _build_stage1_pool_stats(
        pool_ids: List[int],
    ) -> Tuple[Dict[int, int], Dict[int, float], Dict[int, float]]:
        vote_counts_map: Dict[int, int] = defaultdict(int)
        stage1_sum_map: Dict[int, float] = defaultdict(float)
        stage1_peak_map: Dict[int, float] = defaultdict(float)
        pool_set = set(int(pid) for pid in pool_ids)
        for rec in stage1_per_sample:
            rec_scores = rec.get("scores") or []
            sample_candidates = rec.get("candidate_services") or []
            sample_seen: Set[int] = set()
            for pid in sample_candidates:
                try:
                    pid_i = int(pid)
                except Exception:
                    continue
                if pid_i not in pool_set or pid_i in sample_seen:
                    continue
                sample_seen.add(pid_i)
                val = float(rec_scores[pid_i]) if 0 <= pid_i < len(rec_scores) else 0.0
                vote_counts_map[pid_i] += 1
                stage1_sum_map[pid_i] += val
                stage1_peak_map[pid_i] = max(stage1_peak_map[pid_i], val)
        return vote_counts_map, stage1_sum_map, stage1_peak_map

    reranker_pool = _build_union_reranker_pool()
    reranker_diag["candidate_pool_size"] = len(reranker_pool)
    reranker_diag["candidate_pool_source"] = "global_plus_per_sample_union"
    if reranker_dump_path and stage1_per_sample and groundtruths and reranker_pool:
        try:
            os.makedirs(os.path.dirname(reranker_dump_path), exist_ok=True)
            pool_ids = [int(pid) for pid, _ in reranker_pool]
            pool_score_map = {int(pid): float(score) for pid, score in reranker_pool}
            vote_counts_map, stage1_sum_map, stage1_peak_map = _build_stage1_pool_stats(
                pool_ids
            )

            def _gt_pod_set(gt_item) -> Set[int]:
                result: Set[int] = set()
                if gt_item is None:
                    return result
                items = (
                    list(gt_item)
                    if isinstance(gt_item, (list, set, tuple))
                    else [gt_item]
                )
                for it in items:
                    if isinstance(it, dict):
                        ntype_val = str(it.get("ntype", "")).strip().lower()
                        pid_val = it.get("id", it.get("pod_id"))
                        if ntype_val and ntype_val != "pod":
                            continue
                        try:
                            result.add(int(pid_val))
                        except Exception:
                            continue
                    elif isinstance(it, (list, tuple)) and len(it) >= 2:
                        try:
                            if str(it[0]).strip().lower() != "pod":
                                continue
                            result.add(int(it[1]))
                        except Exception:
                            continue
                    else:
                        try:
                            result.add(int(it))
                        except Exception:
                            continue
                return result

            with open(reranker_dump_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "sample_idx",
                        "pod_id",
                        "y",
                        "stage2_score",
                        "stage1_score_sample",
                        "stage1_mean_score",
                        "stage1_vote_count",
                        "stage1_peak_score",
                    ],
                )
                writer.writeheader()
                for sample_idx, rec in enumerate(stage1_per_sample):
                    rec_scores = rec.get("scores") or []
                    gt_set = _gt_pod_set(
                        groundtruths[sample_idx]
                        if sample_idx < len(groundtruths)
                        else None
                    )
                    for pid in pool_ids:
                        stage1_sample = (
                            float(rec_scores[pid])
                            if 0 <= pid < len(rec_scores)
                            else 0.0
                        )
                        votes = vote_counts_map.get(pid, 0)
                        stage1_mean = float(stage1_sum_map.get(pid, 0.0)) / max(
                            votes, 1
                        )
                        writer.writerow(
                            {
                                "sample_idx": sample_idx,
                                "pod_id": pid,
                                "y": 1 if pid in gt_set else 0,
                                "stage2_score": pool_score_map.get(pid, 0.0),
                                "stage1_score_sample": stage1_sample,
                                "stage1_mean_score": stage1_mean,
                                "stage1_vote_count": votes,
                                "stage1_peak_score": float(
                                    stage1_peak_map.get(pid, 0.0)
                                ),
                            }
                        )
            reranker_diag["dump_path"] = reranker_dump_path
        except Exception as e:
            reranker_diag["dump_error"] = str(e)
            logger.warning(f"[Stage 2 reranker] failed to dump training data: {e}")
    if (
        enable_stage2_reranker
        and reranker_model_path
        and reranker_pool
        and stage1_per_sample
    ):
        try:
            import joblib

            model = joblib.load(reranker_model_path)
            pool_ids = [int(pid) for pid, _ in reranker_pool]
            pool_score_map = {int(pid): float(score) for pid, score in reranker_pool}
            vote_counts_map, stage1_sum_map, stage1_peak_map = _build_stage1_pool_stats(
                pool_ids
            )
            rows: List[List[float]] = []
            row_pids: List[int] = []
            row_sample_indices: List[int] = []
            for sample_idx, rec in enumerate(stage1_per_sample):
                rec_scores = rec.get("scores") or []
                for pid in pool_ids:
                    stage1_sample = (
                        float(rec_scores[pid]) if 0 <= pid < len(rec_scores) else 0.0
                    )
                    votes = vote_counts_map.get(pid, 0)
                    stage1_mean = float(stage1_sum_map.get(pid, 0.0)) / max(votes, 1)
                    rows.append(
                        [
                            pool_score_map.get(pid, 0.0),
                            stage1_sample,
                            stage1_mean,
                            float(votes),
                            float(stage1_peak_map.get(pid, 0.0)),
                        ]
                    )
                    row_pids.append(pid)
                    row_sample_indices.append(sample_idx)
            if rows:
                X = np.asarray(rows, dtype=np.float32)
                hybrid_model = (
                    isinstance(model, dict)
                    and model.get("scoring") == "rrf_top1_then_enhanced"
                )
                direct_pack_model = (
                    isinstance(model, dict) and model.get("scoring") == "direct_predict"
                )
                if hybrid_model:
                    if (
                        _stage2_prepare_feature_frame is None
                        or _stage2_assign_hybrid_scores is None
                    ):
                        raise RuntimeError("Hybrid reranker helpers are unavailable.")
                    feature_df = pd.DataFrame(
                        rows,
                        columns=list(STAGE2_RERANKER_BASE_FEATURE_COLS),
                    )
                    feature_df.insert(0, "pod_id", row_pids)
                    feature_df.insert(0, "sample_idx", row_sample_indices)
                    feature_df.insert(2, "y", 0)
                    feature_df, _ = _stage2_prepare_feature_frame(
                        feature_df, "enhanced"
                    )
                    base_cols = list(
                        model.get("base_feature_cols")
                        or STAGE2_RERANKER_BASE_FEATURE_COLS
                    )
                    enhanced_cols = list(model.get("enhanced_feature_cols") or [])
                    missing_cols = [
                        c for c in enhanced_cols if c not in feature_df.columns
                    ]
                    if missing_cols:
                        raise ValueError(
                            f"Hybrid reranker feature columns missing during inference: {missing_cols[:8]}"
                        )
                    feature_df["base_pred_score"] = model["base_model"].predict(
                        feature_df[base_cols].to_numpy(dtype=np.float32)
                    )
                    feature_df["enhanced_pred_score"] = model["enhanced_model"].predict(
                        feature_df[enhanced_cols].to_numpy(dtype=np.float32)
                    )
                    feature_df = _stage2_assign_hybrid_scores(feature_df)
                    pred = feature_df["pred_score"].to_numpy(dtype=np.float32)
                elif direct_pack_model:
                    feature_set = str(model.get("feature_set") or "base")
                    if feature_set == "base":
                        feature_df = pd.DataFrame(
                            rows, columns=list(STAGE2_RERANKER_BASE_FEATURE_COLS)
                        )
                    else:
                        if _stage2_prepare_feature_frame is None:
                            raise RuntimeError(
                                "Enhanced reranker helper is unavailable."
                            )
                        feature_df = pd.DataFrame(
                            rows, columns=list(STAGE2_RERANKER_BASE_FEATURE_COLS)
                        )
                        feature_df.insert(0, "pod_id", row_pids)
                        feature_df.insert(0, "sample_idx", row_sample_indices)
                        feature_df.insert(2, "y", 0)
                        feature_df, _ = _stage2_prepare_feature_frame(
                            feature_df, feature_set
                        )
                    feature_cols = list(
                        model.get("feature_cols") or STAGE2_RERANKER_BASE_FEATURE_COLS
                    )
                    pred = model["model"].predict(
                        feature_df[feature_cols].to_numpy(dtype=np.float32)
                    )
                elif hasattr(model, "predict_proba"):
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X)
                pred = np.asarray(pred, dtype=np.float32).reshape(-1)
                base_vals = th.tensor(
                    [pool_score_map.get(pid, 0.0) for pid in pool_ids], dtype=th.float32
                )
                blend = float(max(0.0, min(1.0, reranker_blend)))
                pred_by_sample_pid: Dict[int, Dict[int, float]] = defaultdict(dict)
                for row_idx, pid in enumerate(row_pids):
                    sample_idx = (
                        row_sample_indices[row_idx]
                        if row_idx < len(row_sample_indices)
                        else 0
                    )
                    pred_by_sample_pid[int(sample_idx)][int(pid)] = float(pred[row_idx])
                pred_scaled_by_sample: List[Dict[int, float]] = []
                for sample_idx in range(len(stage1_per_sample)):
                    sample_pred = pred_by_sample_pid.get(sample_idx, {})
                    pred_vals = th.tensor(
                        [sample_pred.get(pid, 0.0) for pid in pool_ids],
                        dtype=th.float32,
                    )
                    p_min = float(pred_vals.min().item())
                    p_max = float(pred_vals.max().item())
                    if p_max - p_min < 1e-12:
                        pred_scaled = base_vals.clone()
                    else:
                        pred_norm = _safe_minmax_norm(pred_vals)
                        if pred_norm is None:
                            pred_scaled = base_vals.clone()
                        else:
                            b_min = float(base_vals.min().item())
                            b_max = float(base_vals.max().item())
                            span = max(b_max - b_min, 1e-12)
                            pred_scaled = b_min + pred_norm * span
                    pred_scaled_by_sample.append(
                        {
                            int(pool_ids[j]): float(pred_scaled[j].item())
                            for j in range(len(pool_ids))
                        }
                    )
                pred_scaled = th.tensor(
                    [
                        float(
                            np.mean(
                                [
                                    sample_map.get(pid, 0.0)
                                    for sample_map in pred_scaled_by_sample
                                ]
                            )
                        )
                        for pid in pool_ids
                    ],
                    dtype=th.float32,
                )
                blended = (1.0 - blend) * base_vals + blend * pred_scaled
                sorted_idx = sorted(
                    range(len(pool_ids)), key=lambda j: float(blended[j]), reverse=True
                )
                reranked_pool = [
                    (int(pool_ids[j]), float(blended[j])) for j in sorted_idx
                ]
                tail = [
                    (int(pid), float(score))
                    for pid, score in final_shared_ranking
                    if int(pid) not in set(pool_ids)
                ]
                lgbm_shared_top = (reranked_pool + tail)[:top_k_pods]
                transformed_pre_rerank = list(transformed_pod_scores)
                base_stage2_full = {
                    int(pid): float(sc) for pid, sc in transformed_pre_rerank
                }
                ta_ps, ta_sh = _failure_type_aware_rerank(
                    per_sample_final=per_sample_pre_type_aware,
                    stage1_per_sample=stage1_per_sample,
                    labels=labels,
                    top_k_pods=top_k_pods,
                    shared_stage2_scores=transformed_pre_rerank,
                    source_score_map=source_score_map,
                    default_stage1_weight=stage2_stage1_weight,
                    default_stage2_weight=stage2_stage2_weight,
                    default_source_weight=source_score_weight,
                    default_shared_penalty=stage2_shared_candidate_penalty,
                    keep_top1_anchor=stage2_keep_top1_anchor,
                    anchor_margin=stage2_anchor_margin,
                    code_top1_challenger_enabled=code_top1_challenger_enabled,
                    code_top1_challenger_topn=code_top1_challenger_topn,
                    code_top1_challenger_margin=code_top1_challenger_margin,
                    global_hot_challenger_enabled=global_hot_challenger_enabled,
                    global_hot_challenger_topn=global_hot_challenger_topn,
                    global_hot_challenger_threshold=global_hot_challenger_threshold,
                    global_hot_challenger_margin=global_hot_challenger_margin,
                    global_hot_challenger_evidence_margin=global_hot_challenger_evidence_margin,
                    stage1_guard_enabled=stage1_guard_enabled,
                    stage1_guard_topn=stage1_guard_topn,
                    stage1_guard_bonus=stage1_guard_bonus,
                    stage1_guard_min_norm=stage1_guard_min_norm,
                    stage1_guard_gap=stage1_guard_gap,
                    stage1_guard_preserve_top1=stage1_guard_preserve_top1,
                    dynamic_confidence_fusion_enabled=dynamic_confidence_fusion_enabled,
                    dynamic_confidence_strength=dynamic_confidence_strength,
                    dynamic_confidence_gap_topn=dynamic_confidence_gap_topn,
                    dynamic_confidence_max_shift=dynamic_confidence_max_shift,
                    lgbm_rerank_base_stage2=base_stage2_full,
                    lgbm_rerank_pred_scaled=pred_scaled_by_sample,
                    lgbm_rerank_default_blend=blend,
                    lgbm_rerank_replace_body_blend=reranker_blend_replace_body,
                )
                if ta_ps:
                    per_sample_final = ta_ps
                final_shared_ranking = ta_sh[:top_k_pods] if ta_sh else lgbm_shared_top
                reranker_diag.update(
                    {
                        "enabled": True,
                        "model_path": reranker_model_path,
                        "blend": blend,
                        "blend_replace_body": reranker_blend_replace_body,
                        "blend_replace_body_default_if_none": 0.35,
                        "mode": (
                            "hybrid_model"
                            if hybrid_model
                            else ("pack_model" if direct_pack_model else "model")
                        ),
                        "pool_ids": pool_ids,
                        "per_sample_resync": "type_aware_per_sample_lgbm_blend",
                    }
                )
        except Exception as e:
            reranker_diag["error"] = str(e)
            logger.warning(f"[Stage 2 reranker] failed, keep mainline ranking: {e}")
    results = {
        "stage1": {
            "candidate_services": S_cand,
            "tau_star": tau_star,
            "P_prior": P_prior,
            "per_sample_results": stage1_per_sample,
        },
        "stage2": {
            "candidate_pods": final_shared_ranking,
            "all_pod_scores": transformed_pod_scores,
            "raw_all_pod_scores": pruned_pod_scores,
            "candidate_pool": candidate_pool,
            "shared_stage1_scores": stage1_vals,
            "shared_stage2_scores": stage2_vals,
            "shared_source_scores": source_vals,
            "source_score_map": source_score_map,
            "source_diagnostics": source_diag_map,
            "stage2_transform": stage2_transform_diag,
            "vote_counts": vote_counts,
            "shared_diagnostics": {
                "top_candidate_pool": candidate_pool[:top_k_pods],
                "top_shared_ranking": final_shared_ranking[:top_k_pods],
                "stage2_transform": stage2_transform_diag,
                "top_shared_breakdown": [
                    {
                        "pod_id": int(pid),
                        "shared_stage1_score": (
                            float(stage1_vals[idx]) if idx < len(stage1_vals) else 0.0
                        ),
                        "shared_stage2_score": (
                            float(stage2_vals[idx]) if idx < len(stage2_vals) else 0.0
                        ),
                        "shared_source_score": (
                            float(source_vals[idx]) if idx < len(source_vals) else 0.0
                        ),
                        "vote_count": (
                            int(vote_counts[idx]) if idx < len(vote_counts) else 0
                        ),
                        "source_diagnostics": source_diag_map.get(int(pid), {}),
                    }
                    for idx, (pid, _) in enumerate(final_shared_ranking[:top_k_pods])
                ],
            },
            "reranker": reranker_diag,
        },
        "final_ranking": final_shared_ranking,
        "per_sample_results": per_sample_final,
    }
    logger.info("=" * 60)
    logger.info("Lag-aware dual-stage RCA completed")
    logger.info("=" * 60)
    return results


def lag_aware_dual_stage_rca(
    graphs: Union[DGLGraph, List[DGLGraph]],
    stacked_nfeat: Dict[str, th.Tensor],
    data_stats: Dict[str, Dict[str, th.Tensor]],
    labels: List[Dict],
    device: str = "cpu",
    tau_max: int = 3,
    top_k_services: int = 10,
    top_k_pods: int = 10,
    service_timeseries: Optional[th.Tensor] = None,
    service_list: Optional[List[str]] = None,
    G_trace_edges: Optional[List[Tuple[str, str]]] = None,
    propagation_lambda: float = 0.6,
    **kwargs,
) -> Dict:
    return _lag_aware_dual_stage_rca_active(
        graphs=graphs,
        stacked_nfeat=stacked_nfeat,
        data_stats=data_stats,
        labels=labels,
        device=device,
        tau_max=tau_max,
        top_k_services=top_k_services,
        top_k_pods=top_k_pods,
        service_timeseries=service_timeseries,
        service_list=service_list,
        G_trace_edges=G_trace_edges,
        propagation_lambda=propagation_lambda,
        **kwargs,
    )
