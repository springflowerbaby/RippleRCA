import os
import json
import pickle
import random
from datetime import datetime
import pandas as pd
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing._data import _handle_zeros_in_scale
from log import Logger

logger = Logger(__name__)


def set_random_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def plot_line(scores, threshold, filepath):
    legend_labels = []
    if isinstance(scores, np.ndarray) and scores.ndim == 2:
        num_points = scores.shape[0]
        num_nodes = scores.shape[1]
        plt.figure(figsize=(num_points / 50, 8))
        colors = plt.cm.rainbow(np.linspace(0, 1, num_nodes // 4))
        for i in range(num_nodes):
            plt.plot(scores[:, i], label=f"Node {i}", color=colors[i // 4])
            legend_labels.append(f"Node {i}")
    else:
        num_points = len(scores)
        plt.figure(figsize=(num_points / 50, 8))
        plt.plot(scores, label="Scores")
    if threshold is not None:
        plt.axhline(y=threshold, color="red", linestyle="--", label="Threshold")
    plt.xlabel("Index")
    plt.ylabel("Scores")
    plt.title(f"Threshold:{threshold}.")
    plt.legend(legend_labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5)
    plt.tight_layout()
    plt.savefig(filepath)


def get_cur_time():
    return datetime.now().strftime("%Y_%m_%d_%H_%M")


def tensor_to_csv(tensor, prefix, store_dir, columns=None):
    tensor = tensor.cpu()
    df = pd.DataFrame(data=tensor.numpy(), columns=columns)
    df.to_csv(os.path.join(store_dir, prefix + ".csv"))


def graph_to_csv(g, store_dir, suffix):
    dfs = []
    for etype in g.canonical_etypes:
        edges = g.edges(etype=etype)
        df = pd.DataFrame({"src": edges[0].tolist(), "dst": edges[1].tolist()})
        df["etype"] = [etype for _ in range(len(df))]
        dfs.append(df)
    pd.concat(dfs).to_csv(
        os.path.join(store_dir, "graph_{}.csv".format(suffix)), index=False
    )


def stats_to_json(stats, filepath):
    stats_copy = {}
    for name in stats.keys():
        stats_copy[name] = {}
        for ntype in stats[name].keys():
            stats_copy[name][ntype] = stats[name][ntype].tolist()
    with open(filepath, "w") as f:
        json.dump(stats_copy, f)


def load_stats(filepath, device):
    if not os.path.exists(filepath):
        return None
    if filepath.endswith("json"):
        with open(filepath, "r") as f:
            stats = json.load(f)
        for name in stats.keys():
            for ntype in stats[name].keys():
                stats[name][ntype] = th.tensor(stats[name][ntype]).to(device)
    return stats


def dict_to_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logger.info("Save to: {}。".format(filepath))


class myMinMaxScaler(MinMaxScaler):
    def __init__(
        self,
        feature_range: "tuple[int, int]" = (0, 1),
        *,
        copy: bool = True,
        clip: bool = False,
    ):
        super().__init__(feature_range, copy=copy, clip=clip)

    def set_scale_params(self, data_min, data_max):
        feature_range = self.feature_range
        data_range = data_max - data_min
        self.scale_ = (feature_range[1] - feature_range[0]) / _handle_zeros_in_scale(
            data_range, copy=True
        )
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_max_ = data_max
        self.data_min_ = data_min
        self.data_range_ = data_range


def save_node_scalers(scalers, store_dir, type="nodewise"):
    if type not in ("global", "nodewise"):
        return
    with open(os.path.join(store_dir, "node_scalers_{}.pkl".format(type)), "wb") as f:
        pickle.dump(scalers, f)


def save_edge_scalers(scalers, store_dir, type="edgewise"):
    if type not in ("global", "edgewise"):
        return
    with open(os.path.join(store_dir, "edge_scalers_{}.pkl".format(type)), "wb") as f:
        pickle.dump(scalers, f)


def load_node_scalers(store_dir, type="nodewise"):
    with open(os.path.join(store_dir, "node_scalers_{}.pkl".format(type)), "rb") as f:
        scalers = pickle.load(f)
    return scalers


def load_edge_scalers(store_dir, type="edgewise"):
    with open(os.path.join(store_dir, "edge_scalers_{}.pkl".format(type)), "rb") as f:
        scalers = pickle.load(f)
    return scalers


def mahalanobis(u, v, cov_inv, direction: str = "both"):
    delta = u - v
    if direction == "gt":
        delta = th.clamp(delta, min=0)
    elif direction == "lt":
        delta = th.clamp(delta, max=0)
    if len(cov_inv.shape) == 2:
        if u.dim() == v.dim():
            m = th.matmul(delta, th.matmul(cov_inv, delta))
            return th.sqrt(m)
    elif len(cov_inv.shape) == 3:
        ms = []
        for i, slice in enumerate(delta):
            m = th.matmul(slice, th.matmul(cov_inv[i], slice))
            ms.append(th.sqrt(m).item())
        return th.tensor(ms)


def euclidean(u, v, direction: str = "both"):
    delta = u - v
    if direction == "gt":
        delta = th.clamp(delta, min=0)
    elif direction == "lt":
        delta = th.clamp(delta, max=0)
    return th.sqrt(th.sum(th.pow(delta, 2), dim=-1))


def tsne_visualization(
    X: np.array,
    labels: dict,
    title: str = "t-SNE Visualization of Tensor Data with Labels",
    path="images/tmp.png",
):
    tsne = TSNE(n_components=2, random_state=42, method="barnes_hut")
    X_embedded = tsne.fit_transform(X)
    unique_labels = np.unique(list(labels.values()))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {k: v for k, v in zip(unique_labels, colors)}
    print("number of colors: {}".format(len(unique_labels)))
    colors = []
    for k in sorted(labels.keys()):
        colors.append(label_to_color[labels[k]])
    colors = colors * int(X.shape[0] / len(colors))
    plt.figure(figsize=(15, 8))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors)
    for label, color in label_to_color.items():
        plt.scatter([], [], c=[color], label=label)
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(loc="upper right", bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.savefig(path)
