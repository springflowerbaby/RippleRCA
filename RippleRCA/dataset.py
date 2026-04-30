import os
import json
import pickle
from datetime import datetime
from copy import deepcopy as dp
from collections import defaultdict

import dgl
import torch as th
import numpy as np
import pandas as pd
import hashlib
from tqdm import tqdm
from typing import List, Dict, Callable, Optional, Union, Tuple
from dgl.data import DGLDataset
from sklearn.preprocessing import MinMaxScaler

from mask import mask_prepare
from log import Logger

logger = Logger(__name__)


class myDGLDataset(DGLDataset):

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        dates: List[str],
        node_feature_selector: Dict[str, List[str]],
        # edge_feature_selector: Dict[str, List[str]],
        edge_reverse: bool = False,
        add_self_loop: bool = False,
        max_samples: int = 1e9,
        failure_types: List[str] = [],
        special_failure_types: List[str] = [],
        failure_duration: int = 5,
        is_mask: bool = False,
        process_miss: str = 'interpolate',
        process_extreme: bool = False,
        k_sigma: int = 3,
        use_split_info: bool = False,
        **kwargs,
    ):
        self.data_dir = data_dir
        self.dates: List[str] = dates
        self.node_feature_selector: Dict[str, List[str]] = node_feature_selector
        # self.edge_feature_selector: Dict[str, List[str]] = edge_feature_selector
        self.ntypes: List[str] = ['api', 'pod']
        self.etypes: List[str] = [('api', 'to', 'api'), ('pod', 'to', 'api')]
        self.edge_reverse: bool = edge_reverse
        self.add_self_loop: bool = add_self_loop
        self.is_mask: bool = is_mask
        self.max_samples: int = max_samples
        self.failure_duration: int = failure_duration
        self.failure_types: List[str] = failure_types
        # data process
        self.scaler_instance = MinMaxScaler(feature_range=(0, 1))
        self.k = k_sigma  # k-sigma to process extreme values
        self.process_extreme: bool = process_extreme
        self.process_miss: str = process_miss
        # is split train & test by file
        self.use_split_info: bool = use_split_info
        self.special_failure_types = special_failure_types
        # data container
        self.graphs: List[dgl.DGLGraph] = []
        self.labels: List[Dict] = []
        self.groundtruths: List[Tuple[str, int]] = []  # (<ntype>, <dgl_nid>)
        self.num_nodes_dict: Dict[str, int] = {}
        self.name_mapping: Dict[Tuple[str, int], str] = {}

        super().__init__(name=dataset_name)

    def dgl_id_to_name(self, dgl_id, ntype):
        return self.name_mapping[(ntype, dgl_id)]

    def pod_to_service(self, pod_name: str):
        pass

    def label_to_groundtruth(self, label: Dict, node_df: pd.DataFrame, edge_df: pd.DataFrame):
        groundtruth = set()

        if label['level'] == 'service':
            service = label['cmdb_id']
            pod_ids = node_df[(node_df['node_type'] == 'pod') &
                              (node_df['node_name'].str.startswith(service))]['node_id'].tolist()
        elif label['level'] == 'pod':
            service = self.pod_to_service(label['cmdb_id'])
            pod_ids = node_df[node_df['node_name'] == label['cmdb_id']]['node_id'].tolist()

        groundtruth = groundtruth.union([('pod', id) for id in pod_ids])

        api_df = node_df[node_df['node_type'] == 'api']
        df = api_df[api_df['node_name'].str.startswith(service)]
        api_ids = df['node_id'].tolist()

        if label['failure_type'] in self.special_failure_types:
            # extra = api_df[(~api_df['node_name'].str.startswith(service)) &
            #                (api_df['node_name'].str.contains(service, case=False))]['node_id'].tolist()
            # api_ids += extra

            extra_api_ids = []
            for api_id in api_ids:
                tmp = edge_df[(edge_df['edge_type'] == 'api2api') & (edge_df['src'] == api_id)]['dst'].tolist()
                extra_api_ids.extend(tmp)
            api_ids.extend(extra_api_ids)

        groundtruth = groundtruth.union([('api', id) for id in api_ids])

        return list(groundtruth)

    def process(self):
        # read split.json
        if self.use_split_info:
            with open(os.path.join(self.data_dir, '..', f'{self._name}_split.json'), 'r') as f:
                split_info = json.load(f)
            inject_timestamps = [fault['timestamp'] for fault in split_info['test']]

        # read faults
        faults = {}
        for date in self.dates:
            fault_file_path = os.path.join(self.data_dir, '..', f'groundtruth_csv/groundtruth-k8s-1-{date}.csv')
            if not os.path.exists(fault_file_path):
                continue
            fault_df = pd.read_csv(fault_file_path)
            for _, row in fault_df.iterrows():
                for offset in range(0, 60 * self.failure_duration + 1, 60):
                    timestamp_at_minute = row['timestamp'] // 60 * 60 + offset
                    faults[timestamp_at_minute] = {
                        'timestamp': row['timestamp'],
                        'level': row['level'],
                        'cmdb_id': row['cmdb_id'],
                        'failure_type': row['failure_type']
                    }

        # read nodes
        num_nodes_dict = {}
        node_df = pd.read_csv(os.path.join(self.data_dir, 'graph_nodes.csv'))
        for ntype in self.ntypes:
            num_nodes_dict[ntype] = len(node_df[node_df['node_type'] == ntype])
        for _, row in node_df.iterrows():
            self.name_mapping[(row['node_type'], row['node_id'])] = row['node_name']
        self.num_nodes_dict = num_nodes_dict

        # read edges
        edge_dict = {}
        edge_df = pd.read_csv(os.path.join(self.data_dir, 'graph_edges.csv'))
        for etype in self.etypes:
            df = edge_df[edge_df['edge_type'] == etype[0] + '2' + etype[-1]]
            src = th.tensor(df['src'].to_numpy())
            dst = th.tensor(df['dst'].to_numpy())
            if self.edge_reverse:
                etype = (etype[-1], etype[-2], etype[-3])
                src, dst = dst, src
            edge_dict[etype] = (src, dst)

        if self.add_self_loop:
            for ntype, num_nodes in num_nodes_dict.items():
                edge_dict[(ntype, 'self', ntype)] = (th.arange(num_nodes), th.arange(num_nodes))
        else:
            nodes_has_in_edge = {ntype: set() for ntype in num_nodes_dict.keys()}
            for etype, value in edge_dict.items():
                nodes_has_in_edge[etype[2]] |= set(value[1].tolist())
            for ntype, num_nodes in num_nodes_dict.items():
                nodes = th.arange(num_nodes)
                mask = th.isin(nodes, th.tensor(list(nodes_has_in_edge[ntype])))
                nodes = nodes[~mask]
                if nodes.shape[0] != 0:
                    edge_dict[(ntype, 'self', ntype)] = (nodes, nodes)

        # build samples
        for date in sorted(self.dates):
            start_timestamp = int(datetime.strptime(date, '%Y-%m-%d').timestamp())
            end_timestamp = start_timestamp + 24 * 60 * 60
            num_graphs_before = len(self.graphs)

            for timestamp_at_minute in tqdm(range(start_timestamp, end_timestamp, 60), desc=f'Date: {date}'):
                if timestamp_at_minute not in faults:
                    label = {'timestamp': timestamp_at_minute, 'level': '', 'cmdb_id': '', 'failure_type': ''}
                else:
                    label = faults[timestamp_at_minute]

                # filter
                if label['failure_type'] not in self.failure_types:
                    continue
                if self.use_split_info and label['timestamp'] // 60 * 60 not in inject_timestamps:
                    continue

                node_feats = {}
                edge_feats = {}
                file_missing = False
                
                # read node feat
                for ntype in self.ntypes:
                    dir = os.path.join(self.data_dir, f'feats/{date}/{ntype}')
                    # Windows 文件名不能包含 ":"，将时间格式中的 ":" 替换为 "-"
                    time_str = datetime.fromtimestamp(timestamp_at_minute).strftime('%Y-%m-%d %H:%M:%S')
                    time_safe = time_str.replace(":", "-")
                    file = f'{ntype}_feats_{time_safe}.csv'
                    file_path = os.path.join(dir, file)
                    
                    if not os.path.exists(file_path):
                        # 文件不存在，使用零值填充
                        num_nodes = num_nodes_dict[ntype]
                        num_feats = len(self.node_feature_selector[ntype])
                        node_feats[ntype] = th.zeros((num_nodes, num_feats)).float()
                        file_missing = True
                        continue
                    
                    try:
                        df = pd.read_csv(file_path)
                        # Ensure we only take features for nodes that exist in the graph
                        num_nodes = num_nodes_dict[ntype]
                        if len(df) > num_nodes:
                            df = df.iloc[:num_nodes]
                        elif len(df) < num_nodes:
                            logger.warning(f'Feature CSV has {len(df)} rows but graph has {num_nodes} nodes for {ntype}. Padding with zeros.')
                            # Pad with zeros if we have fewer rows
                            missing_rows = num_nodes - len(df)
                            padding_df = pd.DataFrame(0, index=range(missing_rows), columns=df.columns)
                            df = pd.concat([df, padding_df], ignore_index=True)
                        node_feats[ntype] = th.tensor(df[self.node_feature_selector[ntype]].values).float()
                    except Exception as e:
                        logger.error(f'Error reading {file_path}: {e}')
                        # 出错时使用零值
                        num_nodes = num_nodes_dict[ntype]
                        num_feats = len(self.node_feature_selector[ntype])
                        node_feats[ntype] = th.zeros((num_nodes, num_feats)).float()
                        file_missing = True

                    # read edge feat
                    dir = os.path.join(self.data_dir, f'feats/{date}/edge')
                    time_str = datetime.fromtimestamp(timestamp_at_minute).strftime('%Y-%m-%d %H:%M:%S')
                    time_safe = time_str.replace(":", "-")
                    file = f'edge_feats_{time_safe}.csv'
                file_path = os.path.join(dir, file)
                
                if not os.path.exists(file_path):
                    # 文件不存在，使用零值填充
                    for etype, (src, dst) in edge_dict.items():
                        num_edges = len(src)
                        edge_feats[etype] = th.zeros((num_edges, 1)).float()
                    file_missing = True
                else:
                    try:
                        df = pd.read_csv(file_path)
                        for etype, (src, dst) in edge_dict.items():
                            src = src.tolist()
                            dst = dst.tolist()
                            if self.edge_reverse:
                                etype_str = etype[-1] + '2' + etype[0]
                                edge_feat_df = df[df['etype'] == etype_str]
                                edge_feat_df = edge_feat_df.rename(columns={'src': 'dst', 'dst': 'src'})
                            else:
                                etype_str = etype[0] + '2' + etype[-1]
                                edge_feat_df = df[df['etype'] == etype_str]
                            edge_feat_df = edge_feat_df.set_index(['src', 'dst'])
                            edge_feat_df = edge_feat_df.reindex(pd.MultiIndex.from_arrays([src, dst], names=['src', 'dst']))
                            edge_feats[etype] = th.tensor(edge_feat_df['count'].values).unsqueeze(dim=1).float()
                    except Exception as e:
                        logger.error(f'Error reading {file_path}: {e}')
                        # 出错时使用零值
                        for etype, (src, dst) in edge_dict.items():
                            num_edges = len(src)
                            edge_feats[etype] = th.zeros((num_edges, 1)).float()
                        file_missing = True
                
                # 如果文件缺失，记录警告但不跳过
                if file_missing:
                    logger.warning(f'Missing feature files for timestamp {timestamp_at_minute} ({time_str}), using zero values.')

                # mask_prepare
                if self.is_mask:
                    mask_prepare(edge_dict, num_nodes_dict, node_feats, edge_feats)

                graph = dgl.heterograph(data_dict=edge_dict, num_nodes_dict=num_nodes_dict)
                graph.ndata['feat'] = node_feats
                graph.edata['feat'] = edge_feats
                self.graphs.append(graph)
                self.labels.append(label)

                if len(self.graphs) >= self.max_samples:
                    break

            logger.info(f'num_graphs_increased: {len(self.graphs) - num_graphs_before}')
            if len(self.graphs) >= self.max_samples:
                break

        # aggregate fault data
        def _aggregate(tensors: List[th.Tensor]) -> th.Tensor:
            stacked_tensors = th.stack(tensors)
            agg_tensor = th.nanmean(stacked_tensors, dim=0)
            return agg_tensor

        new_graphs = []
        new_labels = []
        for i, label in tqdm(enumerate(self.labels), desc='Aggregating fault data for each case...'):
            if label['failure_type'] == '':
                new_graphs.append(self.graphs[i])
                new_labels.append(self.labels[i])
            else:
                if label in new_labels:
                    continue
                graph = dgl.heterograph(data_dict=edge_dict, num_nodes_dict=num_nodes_dict)

                for ntype in graph.ntypes:
                    feat = []
                    for j in range(i, i + self.failure_duration):
                        feat.append(self.graphs[j].ndata['feat'][ntype])
                    feat = _aggregate(feat)
                    graph.ndata['feat'] = {ntype: feat}

                for etype in graph.canonical_etypes:
                    feat = []
                    for j in range(i, i + self.failure_duration + 1):
                        feat.append(self.graphs[j].edata['feat'][etype])
                    feat = _aggregate(feat)
                    graph.edata['feat'] = {etype: feat}

                new_graphs.append(graph)
                new_labels.append(self.labels[i])
        self.graphs = new_graphs
        self.labels = new_labels

        # label to groundtruth
        for label in self.labels:
            if label['failure_type'] == '':
                continue
            groundtruth = self.label_to_groundtruth(label, node_df, edge_df)
            self.groundtruths.append(groundtruth)

        # process extreme and missing values
        if len(self.graphs) == 0:
            logger.error(f'No graphs loaded for dates {self.dates} with failure_types {self.failure_types}. '
                        f'Please check if the test dates have matching failure data or if failure_types are correct.')
            raise ValueError(f'No graphs loaded. Check test dates and failure_types configuration.')
        
        self.get_nan_nodes()
        self.get_nan_edges()

        if self.process_extreme:
            self.process_extreme_values()
        self.process_missing_values()

        assert len(self.graphs) == len(self.labels)
        logger.info(f'Number of graphs: {len(self.graphs)}')

    def scale(
        self,
        node_scalers=None,
        edge_scalers=None,
        attr='feat',
        node_scale_type='nodewise',
        edge_scale_type='edgewise',
        log=True
    ):

        # 先全部取对数
        if log:
            bias = 1
            for g in self.graphs:
                for ntype in g.ntypes:
                    g.ndata[attr] = {ntype: th.log(g.ndata[attr][ntype] + bias)}

        self.node_scale(node_scalers=node_scalers, attr='feat', node_scale_type=node_scale_type)
        self.edge_scale(edge_scalers=edge_scalers, attr='feat', edge_scale_type=edge_scale_type)

        return self.node_scalers, self.edge_scalers

    def node_scale(self, node_scalers=None, attr='feat', node_scale_type='nodewise'):

        feats = defaultdict(list)
        self.node_scalers = defaultdict(list)

        if node_scale_type == 'nodewise':
            for g in self.graphs:
                for ntype, feat in g.ndata[attr].items():
                    feats[ntype].append(feat.numpy())

            for ntype in self.graphs[0].ntypes:
                # stk: (n_nodes, n_samples, n_feats)
                stk = np.stack(feats[ntype], axis=1)
                for i, slice in enumerate(stk):
                    if node_scalers is None:
                        scaler = dp(self.scaler_instance)
                        stk[i] = scaler.fit_transform(slice)
                        self.node_scalers[ntype].append(scaler)
                    else:
                        # 检查索引是否有效，如果无效则创建新的 scaler
                        if ntype in node_scalers and i < len(node_scalers[ntype]):
                            scaler = node_scalers[ntype][i]
                            stk[i] = scaler.transform(slice)
                        else:
                            # 如果索引超出范围或节点类型不存在，创建新的 scaler 并拟合
                            scaler = dp(self.scaler_instance)
                            stk[i] = scaler.fit_transform(slice)
                        self.node_scalers[ntype].append(scaler)

                stk = stk.swapaxes(0, 1)
                for i, g in enumerate(self.graphs):
                    g.ndata[attr] = {ntype: th.tensor(stk[i])}

        elif node_scale_type == 'global':
            for g in self.graphs:
                for ntype, feat in g.ndata[attr].items():
                    feats[ntype].append(feat.numpy())

            if node_scalers is None:
                for ntype, feat in feats.items():
                    feat = np.concatenate(feat)
                    scaler = dp(self.scaler_instance)
                    scaler = scaler.fit(feat)
                    self.node_scalers[ntype] = scaler
            else:
                self.node_scalers = node_scalers

            for g in self.graphs:
                scaled_feats = {}
                for ntype, feat in g.ndata[attr].items():
                    scaled_feats[ntype] = th.tensor(self.node_scalers[ntype].transform(feat.numpy()))
                g.ndata[attr] = scaled_feats

    def edge_scale(self, edge_scalers=None, attr='feat', edge_scale_type='edgewise'):

        feats = {etype: [] for etype in self.graphs[0].canonical_etypes}
        self.edge_scalers = {etype: [] for etype in self.graphs[0].canonical_etypes}

        for g in self.graphs:
            for etype, feat in g.edata[attr].items():
                feats[etype].append(feat.numpy())

        if edge_scale_type == 'edgewise':
            for etype in self.graphs[0].canonical_etypes:
                # stk: (n_nodes, n_samples, n_feats)
                stk = np.stack(feats[etype], axis=1)
                for i, slice in enumerate(stk):
                    if edge_scalers is None or etype not in edge_scalers.keys():
                        scaler = dp(self.scaler_instance)
                        stk[i] = scaler.fit_transform(slice)
                    else:
                        # 检查索引是否有效，如果无效则创建新的 scaler
                        if i < len(edge_scalers[etype]):
                            scaler = edge_scalers[etype][i]
                            stk[i] = scaler.transform(slice)
                        else:
                            # 如果索引超出范围，创建新的 scaler 并拟合
                            scaler = dp(self.scaler_instance)
                            stk[i] = scaler.fit_transform(slice)
                    self.edge_scalers[etype].append(scaler)

                stk = stk.swapaxes(0, 1)
                for i, g in enumerate(self.graphs):
                    g.edata[attr] = {etype: th.tensor(stk[i])}

        elif edge_scale_type == 'global':
            if edge_scalers is None:
                for etype, feat in feats.items():
                    feat = np.concatenate(feat)
                    scaler = dp(self.scaler_instance)
                    scaler.fit(feat)
                    self.edge_scalers[etype] = scaler
            else:
                self.edge_scalers = edge_scalers

            for g in self.graphs:
                scaled_feats = {}
                for etype, feat in g.edata[attr].items():
                    scaled_feats[etype] = th.tensor(self.edge_scalers[etype].transform(feat.numpy()))
                g.edata[attr] = scaled_feats

    def inverse_scale(self, feats, type='nodewise', log=True):
        """将特征反向缩放回原来的尺度。
        Args:
            feats (dict): 单个样本的特征。
        Returns:
            feats_inv (dict): 反缩放后的特征。
        """
        feats_inv = {}

        if type == 'nodewise':
            for ntype, feat in feats.items():
                feat = feat.cpu()
                feats_inv[ntype] = th.tensor([])
                for i in range(feat.shape[0]):
                    inv_t = th.tensor(
                        self.node_scalers[ntype][i].inverse_transform(th.unsqueeze(feat[i], 0).detach().numpy())
                    )
                    feats_inv[ntype] = th.cat((feats_inv[ntype], inv_t))

        elif type == 'global':
            for ntype, feat in feats.items():
                feat = feat.cpu().detach().numpy()
                feats_inv[ntype] = th.from_numpy(self.node_scalers[ntype].inverse_transform(feat))

        # e指数
        if log:
            bias = 1
            for ntype, feat in feats_inv.items():
                feats_inv[ntype] = th.exp(feat) - bias

        return feats_inv

    def get_feats_stats(self):
        if hasattr(self, 'stats') == False:
            feats = defaultdict(list)

            for graph in self.graphs:
                for ntype in graph.ntypes:
                    # (num_nodes, num_feats)
                    feat = graph.ndata['feat'][ntype]
                    feats[ntype].append(feat)

            rst = defaultdict(dict)

            for ntype, feat in feats.items():
                # (num_samples, num_nodes, num_feats)
                stk = th.stack(feat)
                rst['mean'].update({ntype: th.mean(stk, dim=0)})
                rst['std'].update({ntype: th.std(stk, dim=0)})
                rst['median'].update({ntype: th.median(stk, dim=0)[0]})
                stk = th.swapaxes(stk, 0, 1)
                covs = []
                for slice in stk:
                    covs.append(th.cov(slice))
                rst['cov'].update({ntype: th.stack(covs)})
            self.stats = rst

        return self.stats

    def get_labels(self):
        return self.labels

    def get_groundtruths(self):
        return self.groundtruths

    def get_stacked_nfeat(self):
        feats = {}
        for ntype in self.graphs[0].ntypes:
            feats[ntype] = []

            for g in self.graphs:
                feats[ntype].append(g.ndata['feat'][ntype])

            # (num_samples, num_nodes, num_feats)
            feats[ntype] = th.stack(feats[ntype], dim=0)

        return feats

    def get_stacked_efeat(self):
        feats = {}
        for etype in self.graphs[0].canonical_etypes:
            feats[etype] = []

            for g in self.graphs:
                feats[etype].append(g.edata['feat'][etype])

            # (num_samples, num_edges, num_feats)
            feats[etype] = th.stack(feats[etype], dim=0)

        return feats

    def get_nan_nodes(self):
        if hasattr(self, 'nan_nodes'):
            return self.nan_nodes

        self.nan_nodes = {}
        if len(self.graphs) == 0:
            logger.warning('No graphs loaded, returning empty nan_nodes dictionary.')
            # Return empty nan_nodes for each node type
            for ntype in self.ntypes:
                self.nan_nodes[ntype] = th.tensor([])
            return self.nan_nodes

        for ntype in self.graphs[0].ntypes:
            nans = []
            for g in self.graphs:
                # (num_nodes, num_feats)
                data = g.ndata['feat'][ntype]
                nan = th.where(th.isnan(data).all(dim=1), th.tensor(True), th.tensor(False))
                g.ndata['nan'] = {ntype: nan.reshape(-1, 1)}
                nans.append(nan)
            # (num_samples, num_nodes)
            self.nan_nodes[ntype] = th.stack(nans, dim=0)

        return self.nan_nodes

    def get_nan_edges(self):
        if hasattr(self, 'nan_edges'):
            return self.nan_edges

        self.nan_edges = {}
        if len(self.graphs) == 0:
            logger.warning('No graphs loaded, returning empty nan_edges dictionary.')
            # Return empty nan_edges for each edge type
            for etype in self.etypes:
                self.nan_edges[etype] = th.tensor([])
            return self.nan_edges

        for etype in self.graphs[0].canonical_etypes:
            nans = []

            for g in self.graphs:
                # （num_edges, num_feats)
                data = g.edata['feat'][etype]
                nan = th.where(th.isnan(data).all(dim=1), th.tensor(True), th.tensor(False))
                g.edata['nan'] = {etype: nan.reshape(-1, 1)}
                nans.append(nan)
            self.nan_edges[etype] = th.stack(nans, dim=0)

        return self.nan_edges

    def process_missing_values(self):
        kind = self.process_miss
        logger.info('Processing missing values using {}...'.format(kind))

        def interp_numpy(arr):
            isnan = np.isnan(arr)
            if np.all(isnan):
                return np.nan_to_num(arr, nan=0)
            arr[isnan] = np.interp(np.flatnonzero(isnan), np.flatnonzero(~isnan), arr[~isnan])

            return arr

        # padding ndata
        logger.info('Padding ndata...')
        feats = self.get_stacked_nfeat()

        for ntype, data in feats.items():
            data = data.numpy()
            if kind == 'interpolate':
                for i in range(data.shape[1]):
                    for j in range(data.shape[2]):
                        data[:, i, j] = interp_numpy(data[:, i, j])
            elif kind == 'zero':
                data = np.nan_to_num(data, nan=0)
            for i, feat in enumerate(data):
                self.graphs[i].ndata['feat'] = {ntype: th.tensor(feat).float()}

        # padding edata
        logger.info('Padding edata...')
        feats = self.get_stacked_efeat()

        for etype, data in feats.items():
            data = data.numpy()
            if kind == 'interpolate':
                for i in range(data.shape[1]):
                    for j in range(data.shape[2]):
                        data[:, i, j] = interp_numpy(data[:, i, j])
            elif kind == 'zero':
                data = np.nan_to_num(data, nan=0)
            for i, feat in enumerate(data):
                self.graphs[i].edata['feat'] = {etype: th.tensor(feat).float()}

    def process_extreme_values(self):
        logger.info('Processing extreme values using nan...')
        k = self.k

        def detect_and_replace(data, k):
            # (num_samples, num_nodes, num_feats)
            data = data.numpy()
            mean = np.nanmean(data, axis=0)
            std = np.nanstd(data, axis=0)

            for i in range(data.shape[1]):
                for j in range(data.shape[2]):
                    indices = data[:, i, j] > mean[i, j] + std[i, j] * k
                    data[indices, i, j] = np.nan

                    indices = data[:, i, j] < mean[i, j] - std[i, j] * k
                    data[indices, i, j] = np.nan

            return data

        logger.info('Padding ndata...')
        feats = self.get_stacked_nfeat()

        for ntype, data in feats.items():
            data = detect_and_replace(data, k)

            for i, feat in enumerate(data):
                self.graphs[i].ndata['feat'] = {ntype: th.tensor(feat).float()}

        logger.info('Padding edata...')
        feats = self.get_stacked_efeat()

        for etype, data in feats.items():
            data = detect_and_replace(data, k)

            for i, feat in enumerate(data):
                self.graphs[i].edata['feat'] = {etype: th.tensor(feat).float()}

    def generate_hash_id(self):
        # 将所有变量值转换为字符串并连接起来
        dates_str = ''.join(self.dates) if self.dates is not None else ''
        failure_types_str = ''.join(self.failure_types) if self.failure_types is not None else ''
        data_str = ''.join([
            str(self.data_dir),
            dates_str,
            str(self.node_feature_selector),
            # str(self.edge_feature_selector),
            str(self.ntypes),
            str(self.etypes),
            str(self.edge_reverse),
            str(self.add_self_loop),
            str(self.is_mask),
            str(self.max_samples),
            str(self.failure_duration),
            failure_types_str,
            str(self.process_extreme),
            str(self.process_miss),
            str(self.use_split_info)
        ])

        # 使用md5哈希算法生成哈希值
        hash_object = hashlib.md5(data_str.encode())
        return hash_object.hexdigest()

    def has_cache(self):
        # 生成哈希ID
        hash_id = self.generate_hash_id()
        # 检查文件是否存在
        cache_file = f"./cache/{self._name}_cache_{hash_id}.pkl"
        return os.path.exists(cache_file)

    def load(self):
        # 生成哈希ID
        hash_id = self.generate_hash_id()
        # 加载缓存文件
        cache_file = f"./cache/{self._name}_cache_{hash_id}.pkl"
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        (
            self.graphs, self.labels, self.groundtruths, self.num_nodes_dict, self.name_mapping, self.nan_nodes,
            self.nan_edges
        ) = data
        logger.info(f'Loaded cached dataset {cache_file}')

    def save(self):
        # 生成哈希ID
        hash_id = self.generate_hash_id()
        # 确保缓存目录存在
        cache_dir = "./cache"
        os.makedirs(cache_dir, exist_ok=True)
        # 保存数据到缓存文件
        cache_file = f"./cache/{self._name}_cache_{hash_id}.pkl"
        data = (
            self.graphs, self.labels, self.groundtruths, self.num_nodes_dict, self.name_mapping, self.nan_nodes,
            self.nan_edges
        )
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f'Saved to {cache_file}')

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)
