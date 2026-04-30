import os
import json
import pickle
from datetime import datetime

import dgl
import torch as th
import numpy as np
import pandas as pd
import hashlib
from tqdm import tqdm
from typing import List, Dict, Callable, Optional, Union, Tuple

from dataset import myDGLDataset
from mask import mask_prepare
from log import Logger

logger = Logger(__name__)


class DatasetTrainTicket2024(myDGLDataset):

    def __init__(
        self,
        data_dir,
        dates,
        node_feature_selector,
        # edge_feature_selector,
        edge_reverse,
        add_self_loop,
        max_samples,
        failure_types,
        failure_duration,
        is_mask,
        process_miss,
        process_extreme,
        k_sigma,
        use_split_info,
        **kwargs,
    ):
        super().__init__(
            dataset_name='trainticket_2024',
            data_dir=data_dir,
            dates=dates,
            node_feature_selector=node_feature_selector,
            # edge_feature_selector=edge_feature_selector,
            edge_reverse=edge_reverse,
            add_self_loop=add_self_loop,
            max_samples=max_samples,
            failure_types=failure_types,
            special_failure_types=['network_delay', 'code_delay'],
            failure_duration=failure_duration,
            is_mask=is_mask,
            process_miss=process_miss,
            process_extreme=process_extreme,
            k_sigma=k_sigma,
            use_split_info=use_split_info
        )

    def pod_to_service(self, pod_name: str):
        return '-'.join(pod_name.split('-')[:-2])
