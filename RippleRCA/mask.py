import torch as th
import networkx as nx
import pandas as pd
import os
import re


def save_mask_feat(store_dir, mask_feats, prefix):
    for ntype, feat in mask_feats.items():
        df = pd.DataFrame(data=feat.numpy())
        df.to_csv(os.path.join(store_dir, '{}_{}.csv'.format(prefix, ntype)), index=False)


def load_mask_feat(store_dir, prefix):
    mask_feat = {}
    for file in os.listdir(store_dir):
        match = re.search(r'{}_(\w+)\.csv'.format(prefix), file)
        if match:
            ntype = match.group(1)
        else:
            continue
        mask_feat[ntype] = th.from_numpy(pd.read_csv(os.path.join(store_dir, file)).to_numpy()).to(th.float32)
    return mask_feat


def mask_prepare(edge_dict, num_nodes_dict, node_feats, edge_feats):
    # new ntype: masked_api
    num_nodes_dict['masked_api'] = num_nodes_dict['api']
    # new etypes: pod -> masked_api, api -> masked_api
    edge_dict[('pod', 'to', 'masked_api')] = edge_dict[('pod', 'to', 'api')]
    edge_dict[('api', 'to', 'masked_api')] = edge_dict[('api', 'to', 'api')]
    # copy edge feats
    edge_feats[('pod', 'to', 'masked_api')] = edge_feats[('pod', 'to', 'api')]
    edge_feats[('api', 'to', 'masked_api')] = edge_feats[('api', 'to', 'api')]
    # node feats padding
    node_feats['masked_api'] = th.zeros(node_feats['api'].shape)


def mask_operation(dataset, mask_feats=None):
    # mask all apis
    g = dataset.graphs[0]
    mask = {ntype: th.zeros(g.num_nodes(ntype)) for ntype in g.ntypes}
    mask['api'] = th.ones(g.num_nodes('api'))
    for g in dataset.graphs:
        g.ndata['mask'] = mask
        g.ndata['feat'] = {'masked_api': mask_feats['api']}

    return mask_feats


def replace_node_feats(feats, feats_new, ids):
    new_feats = {}
    for ntype, nfeat in feats.items():
        new_feats[ntype] = nfeat.clone().detach()

    for ntype in ids.keys():
        ids_ntype = ids[ntype]
        if feats_new == 0:
            new_feats[ntype][ids_ntype] = 0
        else:
            new_feats[ntype][ids_ntype] = feats_new[ntype][ids_ntype]

    return new_feats
