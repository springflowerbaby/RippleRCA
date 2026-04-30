import os
from typing import Dict, List, Optional

import numpy as np

from log import Logger

logger = Logger(__name__)

try:
    from tigramite.data_processing import DataFrame
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI
except ImportError:
    # 延迟导入失败的提示，实际调用时再抛出
    DataFrame = None
    ParCorr = None
    PCMCI = None


def _fill_nan_with_column_mean(arr: np.ndarray) -> np.ndarray:
    """将二维时间序列中的 NaN 用列均值填充。"""
    col_mean = np.nanmean(arr, axis=0)
    inds = np.where(np.isnan(arr))
    arr[inds] = np.take(col_mean, inds[1])
    return arr


def _select_nodes(stacked_nfeat, nan_nodes, ntype: str, feat_idx: int, max_nodes: Optional[int]):
    """从 stacked_nfeat 中提取 (T, N) 形式的时间序列矩阵。"""
    data = stacked_nfeat[ntype]  # (T, N, F)
    nan_mask = None
    if nan_nodes is not None and ntype in nan_nodes:
        nan_mask = nan_nodes[ntype]  # (T, N)

    # 选择节点
    if max_nodes is not None:
        data = data[:, :max_nodes, :]
        if nan_mask is not None:
            nan_mask = nan_mask[:, :max_nodes]

    # 仅取一个特征维度
    ts = data[:, :, feat_idx].detach().cpu().numpy()

    # 对于全为 nan 的节点，直接过滤掉
    keep = np.ones(ts.shape[1], dtype=bool)
    for j in range(ts.shape[1]):
        column = ts[:, j]
        if np.isnan(column).all():
            keep[j] = False
        elif nan_mask is not None and nan_mask[:, j].all():
            keep[j] = False
    ts = ts[:, keep]

    if ts.size == 0:
        raise ValueError(f'{ntype} 没有可用的节点可用于 PCMCIplus。')

    ts = _fill_nan_with_column_mean(ts)
    return ts, keep


def run_pcmci_plus_from_stacked(
    stacked_nfeat,
    labels: List[Dict],
    ntype: str = 'api',
    feat_idx: int = 0,
    max_nodes: Optional[int] = 50,
    tau_min: int = 1,
    tau_max: int = 3,
    alpha: float = 0.05,
    save_dir: Optional[str] = None,
    dgl_id_to_name=None,
    nan_nodes=None,
):
    """在堆叠特征上运行 PCMCIplus，并返回显著因果边。"""
    if DataFrame is None or ParCorr is None or PCMCI is None:
        raise ImportError('未找到 tigramite，请先安装：pip install tigramite==5.2.9.4')

    ts, keep_mask = _select_nodes(stacked_nfeat, nan_nodes, ntype, feat_idx, max_nodes)
    logger.info(f'PCMCIplus: 使用 {ts.shape[0]} 个时间点、{ts.shape[1]} 个变量，tau_max={tau_max}, alpha={alpha}')

    dataframe = DataFrame(ts)
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

    results = pcmci.run_pcmciplus(tau_min=tau_min, tau_max=tau_max, pc_alpha=alpha)
    val_matrix = results['val_matrix']
    q_matrix = results.get('q_matrix', None)
    p_matrix = results.get('p_matrix', None)

    edges = []
    n_vars = ts.shape[1]
    for to in range(n_vars):
        for frm in range(n_vars):
            for lag in range(tau_min, tau_max + 1):
                val = val_matrix[to, frm, lag]
                p_val = None if p_matrix is None else p_matrix[to, frm, lag]
                q_val = None if q_matrix is None else q_matrix[to, frm, lag]
                sig = q_val if q_val is not None else p_val
                if sig is None or np.isnan(sig) or sig > alpha:
                    continue
                edges.append({
                    'from': frm,
                    'to': to,
                    'lag': lag,
                    'val': float(val),
                    'p_val': None if p_val is None else float(p_val),
                    'q_val': None if q_val is None else float(q_val),
                })

    # 映射回 dgl id，若提供
    if dgl_id_to_name is not None:
        mapped_edges = []
        # keep_mask 标记保留的节点，下标 -> 原节点 id
        original_ids = [i for i, k in enumerate(keep_mask) if k]
        for e in edges:
            frm_id = original_ids[e['from']]
            to_id = original_ids[e['to']]
            mapped_edges.append({
                **e,
                'from_id': frm_id,
                'to_id': to_id,
                'from_name': dgl_id_to_name(frm_id, ntype),
                'to_name': dgl_id_to_name(to_id, ntype),
            })
        edges = mapped_edges

    summary = {
        'ntype': ntype,
        'feat_idx': feat_idx,
        'tau_max': tau_max,
        'alpha': alpha,
        'num_vars': n_vars,
        'num_edges': len(edges),
    }

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        edges_path = os.path.join(save_dir, f'pcmci_plus_edges_{ntype}.json')
        summary_path = os.path.join(save_dir, f'pcmci_plus_summary_{ntype}.json')
        import json

        with open(edges_path, 'w', encoding='utf-8') as f:
            json.dump(edges, f, ensure_ascii=False, indent=2)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        summary['edges_path'] = edges_path
        summary['summary_path'] = summary_path

    return summary, edges

