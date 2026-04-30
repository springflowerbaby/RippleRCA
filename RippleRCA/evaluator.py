"""
评估模块

计算top-1, top-3, top-5准确率和其他RCA指标。
"""

from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import numpy as np  # pyright: ignore[reportMissingImports]

from log import Logger  # pyright: ignore[reportMissingImports]

logger = Logger(__name__)


def compute_rca_metrics(
    rsts_all: List[Dict],
    groundtruths_all: List[Set[Tuple[str, int]]],
    failure_type: Optional[str] = None,
    top_k_list: List[int] = None,
    agg: bool = False,
) -> Dict:
    """
    计算RCA指标：top-k准确率和平均排名。
    
    Args:
        rsts_all: RCA结果列表，每个结果包含 'cand' 字段（候选列表）
        groundtruths_all: 真实根因列表，每个是一个集合 {(ntype, id)}
        failure_type: 故障类型（可选，用于过滤）
        top_k_list: 要计算的top-k列表，默认 [1, 3, 5]
        agg: 是否聚合结果
    
    Returns:
        指标字典，包含:
        - n: 样本数量
        - skipped: 跳过的样本数量
        - top-{k}: top-k准确率
        - avg_rank: 平均排名
        - avg@{max_k}: 平均准确率
    """
    if top_k_list is None:
        top_k_list = [1, 3, 5]
    
    # 过滤故障类型（兼容 AIOps2025：label 中为 network_delay_with_propagation 等，与 config 的 network_delay 做包含匹配）
    if failure_type is None:
        rsts = rsts_all
        groundtruths = groundtruths_all
    else:
        rsts = []
        groundtruths = []
        for i, rst in enumerate(rsts_all):
            lbl_ft = (rst.get('failure_type') or '').strip()
            if lbl_ft == failure_type or (failure_type in lbl_ft):
                rsts.append(rst)
                groundtruths.append(groundtruths_all[i])
    
    # 聚合groundtruths（如果需要）
    if agg:
        # 注意：这里可能需要根据实际需求实现聚合逻辑
        pass
    
    # 计算top-acc和average rank
    max_rank = max(top_k_list)
    top_acc = {k: 0 for k in top_k_list}
    avg_rank = 0.0
    reciprocal_rank_sum = 0.0
    reciprocal_rank_sum_at_max = 0.0
    skipped = 0
    
    for idx, rst in enumerate(rsts):
        groundtruth = groundtruths[idx] if idx < len(groundtruths) else set()
        is_find = False
        
        if 'cand' not in rst.keys() or not rst['cand']:
            skipped += 1
            continue
        
        # 遍历候选列表
        for i, candidate in enumerate(rst['cand']):
            # 处理不同的候选格式
            if isinstance(candidate, tuple):
                # (pod_id, score) 格式
                pod_id = candidate[0]
                # 需要映射到 (ntype, id)
                cand_tuple = ('pod', pod_id)  # 简化假设
            elif isinstance(candidate, dict):
                # {'ntype': 'pod', 'id': 3, 'score': 53.0} 格式
                cand_tuple = (candidate.get('ntype', 'pod'), candidate.get('id'))
            else:
                continue
            
            # 检查是否在真实根因中
            if cand_tuple in groundtruth:
                rank = i + 1
                avg_rank += rank
                reciprocal_rank_sum += 1.0 / rank
                if rank <= max_rank:
                    reciprocal_rank_sum_at_max += 1.0 / rank
                
                # 更新top-k准确率
                for k in top_k_list:
                    if rank <= k:
                        top_acc[k] += 1
                
                is_find = True
                if idx < 3:  # 调试信息
                    logger.debug(f'  [OK] Found groundtruth at rank {rank}: {cand_tuple}')
                break
        
        if not is_find:
            avg_rank += max_rank + 1  # 未找到，使用max_rank+1作为排名
            if idx < 3:  # 调试信息
                logger.debug(f'  [X] Groundtruth not found in top {len(rst["cand"])} candidates')
    
    # 计算有效结果数量
    valid_count = len(rsts) - skipped
    if valid_count == 0:
        logger.warning(f'No valid results found for failure_type: {failure_type}. All results were skipped or empty.')
        metrics = {
            'n': len(rsts),
            'skipped': skipped,
        }
        for k in top_k_list:
            metrics[f'top-{k}'] = 0.0
        metrics['avg_rank'] = 0.0
        metrics['mrr'] = 0.0
        metrics[f'mrr@{max_rank}'] = 0.0
        metrics['mrr_full'] = 0.0
        metrics[f'avg@{max_rank}'] = 0.0
        return metrics
    
    # 归一化指标
    for k in top_k_list:
        top_acc[k] /= valid_count
    avg_rank /= valid_count
    mrr_full = reciprocal_rank_sum / valid_count
    mrr_at_max = reciprocal_rank_sum_at_max / valid_count
    avg_at_max = np.mean([top_acc[k] for k in top_k_list])
    
    # 构建指标字典
    metrics = {
        'n': len(rsts),
        'skipped': skipped,
        'valid_count': valid_count,
    }
    for k in top_k_list:
        metrics[f'top-{k}'] = top_acc[k]
    metrics['avg_rank'] = avg_rank
    # Keep "mrr" aligned with Top-K metrics: misses outside max_rank contribute 0.
    metrics['mrr'] = mrr_at_max
    metrics[f'mrr@{max_rank}'] = mrr_at_max
    metrics['mrr_full'] = mrr_full
    metrics[f'avg@{max_rank}'] = avg_at_max
    
    # 打印结果
    logger.info(f'-----------------RCA metrics: {failure_type}-----------------')
    logger.info(f'n: {len(rsts)}. skipped: {skipped}. valid: {valid_count}.')
    logger.info(f'avg_rank: {avg_rank:.4f}.')
    logger.info(f'mrr@{max_rank}: {mrr_at_max:.4f}.')
    logger.info(f'mrr_full: {mrr_full:.4f}.')
    for k in top_k_list:
        logger.info(f'top-{k} acc: {top_acc[k]:.4f}.')
    logger.info(f'avg@{max_rank}: {avg_at_max:.4f}')
    
    return metrics


def format_results_for_evaluation(
    rca_results: Dict,
    labels: List[Dict],
    per_sample_results: bool = False,
    per_row_graph_indices: Optional[List[int]] = None,
) -> List[Dict]:
    """
    将RCA结果格式化为评估所需的格式。
    
    Args:
        rca_results: lag_aware_dual_stage_rca的返回结果
        labels: 标签列表
        per_sample_results: 是否为每个样本生成单独的结果（如果rca_results包含每个样本的结果）
        per_row_graph_indices: 每行对应的图索引（与 labels 一一对应），用于多行少图时按图取排序
    
    Returns:
        格式化的结果列表，每个元素包含:
        - timestamp
        - failure_type
        - cmdb_id (或其他标识)
        - cand: 候选列表 [{'ntype': 'pod', 'id': id, 'score': score}, ...]
    """
    formatted_results = []
    per_sample = (rca_results.get('per_sample_results') or []) if isinstance(rca_results.get('per_sample_results'), list) else []
    use_per_sample = (per_sample_results or per_row_graph_indices is not None) and len(per_sample) > 0
    
    if use_per_sample:
        # 按样本/按图使用排序：每行用对应图索引的 final_ranking
        for i, label in enumerate(labels):
            if per_row_graph_indices is not None and i < len(per_row_graph_indices):
                gidx = per_row_graph_indices[i]
                if 0 <= gidx < len(per_sample):
                    pod_candidates = per_sample[gidx].get('final_ranking', [])
                else:
                    pod_candidates = rca_results.get('final_ranking', [])
            elif i < len(per_sample):
                pod_candidates = per_sample[i].get('final_ranking', [])
            else:
                pod_candidates = rca_results.get('final_ranking', [])
            
            cand_list = []
            for item in pod_candidates:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    pod_id, score = item[0], item[1]
                elif isinstance(item, dict):
                    pod_id, score = item.get('id', item.get('pod_id')), item.get('score', 0.0)
                else:
                    continue
                cand_list.append({'ntype': 'pod', 'id': pod_id, 'score': float(score)})
            
            formatted_results.append({
                'timestamp': label.get('timestamp'),
                'failure_type': label.get('failure_type'),
                'cmdb_id': label.get('cmdb_id'),
                'cand': cand_list,
            })
    else:
        # 使用全局候选列表
        pod_candidates = rca_results.get('final_ranking', [])
        
        # 将pod_candidates转换为标准格式
        cand_list = []
        for item in pod_candidates:
            if isinstance(item, tuple):
                pod_id, score = item
            elif isinstance(item, dict):
                pod_id = item.get('id', item.get('pod_id'))
                score = item.get('score', 0.0)
            else:
                continue
            
            cand_list.append({
                'ntype': 'pod',
                'id': pod_id,
                'score': score,
            })
        
        # 为每个标签创建结果
        for label in labels:
            formatted_result = {
                'timestamp': label.get('timestamp'),
                'failure_type': label.get('failure_type'),
                'cmdb_id': label.get('cmdb_id'),
                'cand': cand_list.copy(),  # 每个样本使用相同的候选列表
            }
            formatted_results.append(formatted_result)
    
    return formatted_results


def evaluate_lads_results(
    rca_results: Dict,
    groundtruths: List[Set[Tuple[str, int]]],
    labels: List[Dict],
    failure_types: Optional[List[str]] = None,
    top_k_list: List[int] = None,
    per_row_graph_indices: Optional[List[int]] = None,
) -> Dict:

    if top_k_list is None:
        top_k_list = [1, 3, 5]
    if failure_types is None:
        failure_types = []
    
    logger.info('=' * 60)
    logger.info('LADS-Causal 结果评估')
    logger.info('=' * 60)
    
    # 格式化结果（按行评估时传入 per_row_graph_indices，使每行使用对应图的排序）
    formatted_results = format_results_for_evaluation(
        rca_results, labels,
        per_row_graph_indices=per_row_graph_indices,
    )
    
    # 计算整体指标
    overall_metrics = compute_rca_metrics(
        formatted_results, groundtruths,
        failure_type=None,
        top_k_list=top_k_list,
    )
    
    # 按故障类型计算指标
    type_metrics = {}
    for failure_type in failure_types:
        type_metrics[failure_type] = compute_rca_metrics(
            formatted_results, groundtruths,
            failure_type=failure_type,
            top_k_list=top_k_list,
        )
    
    # 汇总结果
    evaluation_results = {
        'overall': overall_metrics,
        'by_failure_type': type_metrics,
        'summary': {
            'total_samples': len(formatted_results),
            'failure_types': failure_types,
            'top_k_list': top_k_list,
        },
    }
    
    logger.info('=' * 60)
    logger.info('评估完成')
    logger.info('=' * 60)
    
    return evaluation_results
