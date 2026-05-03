"""Evaluation helpers for RippleRCA."""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import numpy as np  # pyright: ignore[reportMissingImports]
from log import Logger  # pyright: ignore[reportMissingImports]

logger = Logger(__name__)


def compute_rca_metrics(
    results: List[Dict],
    groundtruths: List[Set[Tuple[str, int]]],
    failure_type: Optional[str] = None,
    top_k_list: Optional[List[int]] = None,
) -> Dict:
    if top_k_list is None:
        top_k_list = [1, 3, 5]
    if failure_type is None:
        filtered_results = results
        filtered_groundtruths = groundtruths
    else:
        filtered_results = []
        filtered_groundtruths = []
        for i, result in enumerate(results):
            label_failure_type = (result.get("failure_type") or "").strip()
            if label_failure_type == failure_type or failure_type in label_failure_type:
                filtered_results.append(result)
                filtered_groundtruths.append(groundtruths[i])
    max_rank = max(top_k_list)
    top_acc = {k: 0 for k in top_k_list}
    avg_rank = 0.0
    reciprocal_rank_sum = 0.0
    reciprocal_rank_sum_at_max = 0.0
    skipped = 0
    for idx, result in enumerate(filtered_results):
        groundtruth = (
            filtered_groundtruths[idx] if idx < len(filtered_groundtruths) else set()
        )
        found = False
        if "cand" not in result or not result["cand"]:
            skipped += 1
            continue
        for i, candidate in enumerate(result["cand"]):
            if isinstance(candidate, tuple):
                candidate_tuple = ("pod", candidate[0])
            elif isinstance(candidate, dict):
                candidate_tuple = (candidate.get("ntype", "pod"), candidate.get("id"))
            else:
                continue
            if candidate_tuple in groundtruth:
                rank = i + 1
                avg_rank += rank
                reciprocal_rank_sum += 1.0 / rank
                if rank <= max_rank:
                    reciprocal_rank_sum_at_max += 1.0 / rank
                for k in top_k_list:
                    if rank <= k:
                        top_acc[k] += 1
                found = True
                break
        if not found:
            avg_rank += max_rank + 1
    valid_count = len(filtered_results) - skipped
    if valid_count == 0:
        metrics = {"n": len(filtered_results), "skipped": skipped}
        for k in top_k_list:
            metrics[f"top-{k}"] = 0.0
        metrics["avg_rank"] = 0.0
        metrics["mrr"] = 0.0
        metrics[f"mrr@{max_rank}"] = 0.0
        metrics["mrr_full"] = 0.0
        metrics[f"avg@{max_rank}"] = 0.0
        return metrics
    for k in top_k_list:
        top_acc[k] /= valid_count
    avg_rank /= valid_count
    mrr_full = reciprocal_rank_sum / valid_count
    mrr_at_max = reciprocal_rank_sum_at_max / valid_count
    avg_at_max = np.mean([top_acc[k] for k in top_k_list])
    metrics = {
        "n": len(filtered_results),
        "skipped": skipped,
        "valid_count": valid_count,
        "avg_rank": avg_rank,
        "mrr": mrr_at_max,
        f"mrr@{max_rank}": mrr_at_max,
        "mrr_full": mrr_full,
        f"avg@{max_rank}": avg_at_max,
    }
    for k in top_k_list:
        metrics[f"top-{k}"] = top_acc[k]
    return metrics


def format_results_for_evaluation(
    rca_results: Dict,
    labels: List[Dict],
    per_row_graph_indices: Optional[List[int]] = None,
) -> List[Dict]:
    formatted_results = []
    per_sample = (
        rca_results.get("per_sample_results") or []
        if isinstance(rca_results.get("per_sample_results"), list)
        else []
    )
    use_per_sample = per_row_graph_indices is not None and len(per_sample) > 0
    if use_per_sample:
        for i, label in enumerate(labels):
            if i < len(per_row_graph_indices):
                graph_idx = per_row_graph_indices[i]
                if 0 <= graph_idx < len(per_sample):
                    pod_candidates = per_sample[graph_idx].get("final_ranking", [])
                else:
                    pod_candidates = rca_results.get("final_ranking", [])
            elif i < len(per_sample):
                pod_candidates = per_sample[i].get("final_ranking", [])
            else:
                pod_candidates = rca_results.get("final_ranking", [])
            cand_list = []
            for item in pod_candidates:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    pod_id, score = item[0], item[1]
                elif isinstance(item, dict):
                    pod_id, score = item.get("id", item.get("pod_id")), item.get(
                        "score", 0.0
                    )
                else:
                    continue
                cand_list.append({"ntype": "pod", "id": pod_id, "score": float(score)})
            formatted_results.append(
                {
                    "timestamp": label.get("timestamp"),
                    "failure_type": label.get("failure_type"),
                    "cmdb_id": label.get("cmdb_id"),
                    "cand": cand_list,
                }
            )
    else:
        pod_candidates = rca_results.get("final_ranking", [])
        cand_list = []
        for item in pod_candidates:
            if isinstance(item, tuple):
                pod_id, score = item
            elif isinstance(item, dict):
                pod_id = item.get("id", item.get("pod_id"))
                score = item.get("score", 0.0)
            else:
                continue
            cand_list.append({"ntype": "pod", "id": pod_id, "score": score})
        for label in labels:
            formatted_results.append(
                {
                    "timestamp": label.get("timestamp"),
                    "failure_type": label.get("failure_type"),
                    "cmdb_id": label.get("cmdb_id"),
                    "cand": cand_list.copy(),
                }
            )
    return formatted_results


def evaluate_results(
    rca_results: Dict,
    groundtruths: List[Set[Tuple[str, int]]],
    labels: List[Dict],
    failure_types: Optional[List[str]] = None,
    top_k_list: Optional[List[int]] = None,
    per_row_graph_indices: Optional[List[int]] = None,
) -> Dict:
    if top_k_list is None:
        top_k_list = [1, 3, 5]
    if failure_types is None:
        failure_types = []
    logger.info("=" * 60)
    logger.info("RippleRCA evaluation")
    logger.info("=" * 60)
    formatted_results = format_results_for_evaluation(
        rca_results, labels, per_row_graph_indices=per_row_graph_indices
    )
    overall_metrics = compute_rca_metrics(
        formatted_results,
        groundtruths,
        failure_type=None,
        top_k_list=top_k_list,
    )
    type_metrics = {}
    for failure_type in failure_types:
        type_metrics[failure_type] = compute_rca_metrics(
            formatted_results,
            groundtruths,
            failure_type=failure_type,
            top_k_list=top_k_list,
        )
    return {
        "overall": overall_metrics,
        "by_failure_type": type_metrics,
        "summary": {
            "total_samples": len(formatted_results),
            "failure_types": failure_types,
            "top_k_list": top_k_list,
        },
    }
