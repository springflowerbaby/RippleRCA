"""
数据处理流水线模块

用于处理TrainTicket原始数据（Trace和Metric），生成LADS-Causal框架所需的数据格式。
包含：
1. Trace解析与动态拓扑构建
2. Metric数据对齐与聚合
3. 训练样本构建
"""

import os
import json
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict

import pandas as pd  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import torch as th  # pyright: ignore[reportMissingImports]
from tqdm import tqdm  # pyright: ignore[reportMissingModuleSource]
import dgl  # pyright: ignore[reportMissingImports]


class TraceParser:
    """Trace解析器 - 从Jaeger/Zipkin导出数据构建服务调用拓扑

    """
    
    def __init__(
        self,
        trace_dir: str,
        min_call_frequency: int = 5,  # 每分钟最少调用次数（过滤低频调用）
        time_window: int = 60,  # 时间窗口（秒），用于统计调用频率
    ):
        """
        Args:
            trace_dir: Trace JSON文件目录
            min_call_frequency: 最小调用频率阈值
            time_window: 时间窗口（秒）
        """
        self.trace_dir = trace_dir
        self.min_call_frequency = min_call_frequency
        self.time_window = time_window
    
    def parse_traces(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], int]]:
        """
        解析Trace数据，构建服务调用关系
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
        
        Returns:
            (trace_df, call_counts): 
            - trace_df: Trace数据DataFrame
            - call_counts: {(parent_service, child_service): count} 调用次数统计
        """
        print(f"解析Trace数据: {self.trace_dir}")
        
        # 支持 JSON 与 CSV
        json_files = [f for f in os.listdir(self.trace_dir) if f.endswith('.json')]
        csv_files = [f for f in os.listdir(self.trace_dir) if f.endswith('.csv')]
        
        if json_files:
            # 原有 JSON 逻辑
            all_traces = []
            for trace_file in tqdm(json_files, desc="读取Trace文件"):
                filepath = os.path.join(self.trace_dir, trace_file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    traces = json.load(f)
                    if isinstance(traces, list):
                        all_traces.extend(traces)
                    else:
                        all_traces.append(traces)
            
            trace_records = []
            for trace in all_traces:
                spans = trace['data'] if isinstance(trace, dict) and 'data' in trace else trace
                spans = spans if isinstance(spans, list) else [spans]
                for span in spans:
                    if isinstance(span, dict):
                        parent_service = span.get('process', {}).get('serviceName', '')
                        operation_name = span.get('operationName', '')
                        start_ts = span.get('startTime', 0) / 1000  # 秒
                        duration = span.get('duration', 0) / 1000
                        child_service = None
                        if 'references' in span:
                            refs = span['references']
                            if refs and len(refs) > 0:
                                child_service = refs[0].get('process', {}).get('serviceName', '')
                        trace_records.append({
                            'timestamp': datetime.fromtimestamp(start_ts),
                            'parent_service': parent_service,
                            'child_service': child_service or '',
                            'operation': operation_name,
                            'duration': duration,
                        })
            trace_df = pd.DataFrame(trace_records)
        
        elif csv_files:
            # TrainTicket_2024 CSV 格式
            dfs = []
            for trace_file in tqdm(csv_files, desc="读取Trace文件"):
                filepath = os.path.join(self.trace_dir, trace_file)
                df = pd.read_csv(filepath)
                dfs.append(df)
            trace_df_raw = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            
            if trace_df_raw.empty:
                return pd.DataFrame(), {}
            
            max_rows = 200_000
            if len(trace_df_raw) > max_rows:
                print(f"Trace 数据行数为 {len(trace_df_raw)}，采样为 {max_rows} 行以控制内存占用")
                trace_df_raw = trace_df_raw.sample(n=max_rows, random_state=42)
            
            # 构造 span_id -> pod 映射
            span_to_pod = {}
            if 'SpanID' in trace_df_raw.columns and 'PodName' in trace_df_raw.columns:
                span_to_pod = dict(zip(trace_df_raw['SpanID'], trace_df_raw['PodName']))
            
            def pod_to_service(pod: str) -> str:
                parts = pod.split('-')
                return '-'.join(parts[:-2]) if len(parts) > 2 else pod
            
            records = []
            for _, row in trace_df_raw.iterrows():
                # 计算时间戳
                if 'StartTimeUnixNano' in row:
                    ts = pd.to_datetime(int(row['StartTimeUnixNano']), unit='ns')
                elif 'StartTime' in row:
                    ts = pd.to_datetime(row['StartTime'])
                else:
                    continue
                
                child_pod = row.get('PodName', '')
                child_service = pod_to_service(child_pod) if child_pod else ''
                
                parent_service = ''
                parent_id = row.get('ParentID', '')
                if isinstance(parent_id, str) and parent_id and parent_id != 'root':
                    parent_pod = span_to_pod.get(parent_id, '')
                    parent_service = pod_to_service(parent_pod) if parent_pod else ''
                
                records.append({
                    'timestamp': ts.to_pydatetime(),
                    'parent_service': parent_service,
                    'child_service': child_service,
                    'operation': row.get('OperationName', ''),
                    'duration': row.get('Duration', 0),
                })
            
            trace_df = pd.DataFrame(records)
        else:
            # 无文件
            print("未找到 trace 文件")
            return pd.DataFrame(), {}
        
        # 过滤时间范围
        if start_time:
            trace_df = trace_df[trace_df['timestamp'] >= start_time]
        if end_time:
            trace_df = trace_df[trace_df['timestamp'] <= end_time]
        
        if trace_df.empty:
            print("Trace 数据为空")
            return trace_df, {}
        
        # 按时间窗口统计调用次数
        trace_df['time_window'] = trace_df['timestamp'].dt.floor(f'{self.time_window}s')
        
        # 统计服务调用关系
        call_counts = defaultdict(int)
        for _, row in trace_df.iterrows():
            if row['parent_service'] and row['child_service']:
                key = (row['parent_service'], row['child_service'])
                call_counts[key] += 1
        
        # 过滤低频调用
        filtered_call_counts = {
            k: v for k, v in call_counts.items()
            if v >= self.min_call_frequency
        }
        
        print(f"发现 {len(filtered_call_counts)} 个服务调用关系")
        
        return trace_df, dict(filtered_call_counts)
    
    def build_adjacency_matrix(
        self,
        call_counts: Dict[Tuple[str, str], int],
        service_list: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        构建邻接矩阵
        
        Args:
            call_counts: 服务调用统计
            service_list: 服务列表（如果为None，则从call_counts中提取）
        
        Returns:
            (adjacency_matrix, service_list): 邻接矩阵和服务列表
        """
        if service_list is None:
            # 从call_counts中提取所有服务
            all_services = set()
            for parent, child in call_counts.keys():
                all_services.add(parent)
                all_services.add(child)
            service_list = sorted(list(all_services))
        
        n_services = len(service_list)
        service_to_idx = {svc: i for i, svc in enumerate(service_list)}
        
        # 构建邻接矩阵
        adj_matrix = np.zeros((n_services, n_services), dtype=np.float32)
        
        for (parent, child), count in call_counts.items():
            if parent in service_to_idx and child in service_to_idx:
                parent_idx = service_to_idx[parent]
                child_idx = service_to_idx[child]
                adj_matrix[parent_idx, child_idx] = count
        
        return adj_matrix, service_list
    
    def get_trace_topology(
        self,
        call_counts: Dict[Tuple[str, str], int]
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        从 call_counts 中提取服务列表和边列表，用于 Stage 1 的约束图 G_trace
        
        Args:
            call_counts: 服务调用统计 {(parent_service, child_service): count}
        
        Returns:
            (service_list, edge_list):
            - service_list: 排序后的服务列表
            - edge_list: 边列表 [(u, v), ...]，表示存在调用关系的边
        """
        # 提取所有服务
        all_services = set()
        for parent, child in call_counts.keys():
            if parent:  # 过滤空字符串
                all_services.add(parent)
            if child:
                all_services.add(child)
        
        service_list = sorted(list(all_services))
        
        # 提取边列表（只包含有效的调用关系）
        edge_list = [
            (parent, child) for (parent, child) in call_counts.keys()
            if parent and child
        ]
        
        return service_list, edge_list


class MetricAggregator:
    """Metric数据聚合器 - 从Prometheus数据生成服务级和Pod级特征"""
    
    def __init__(
        self,
        metric_dir: str,
        sampling_interval: int = 10,  # 采样间隔（秒）
    ):
        """
        Args:
            metric_dir: Metric数据目录
            sampling_interval: 重采样间隔（秒）
        """
        self.metric_dir = metric_dir
        self.sampling_interval = sampling_interval
    
    def load_prometheus_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        加载Prometheus指标数据
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
        
        Returns:
            Metric DataFrame，包含以下列：
            - timestamp
            - service_name (或 pod_name)
            - metric_name
            - metric_value
        """
        print(f"加载Prometheus指标: {self.metric_dir}")
        
        if not os.path.isdir(self.metric_dir):
            print(f"[WARN] Metric 目录不存在: {self.metric_dir}，返回空 DataFrame")
            return pd.DataFrame(columns=["timestamp", "PodName", "service"])
        
        metric_files = [f for f in os.listdir(self.metric_dir) if f.endswith('.csv')]
        
        all_metrics = []
        for metric_file in tqdm(metric_files, desc="读取Metric文件"):
            filepath = os.path.join(self.metric_dir, metric_file)
            df = pd.read_csv(filepath)
            all_metrics.append(df)
        
        if not all_metrics:
            print(f"[WARN] 未找到 Metric CSV 文件: {self.metric_dir}，返回空 DataFrame（将使用原有方法）")
            return pd.DataFrame(columns=["timestamp", "PodName", "service"])
        
        metric_df = pd.concat(all_metrics, ignore_index=True)
        
        # 转换时间戳（兼容 TrainTicket / AIOps2025：timestamp, time, TimeStamp, datetime, t, ts 等）
        time_col = None
        for col in ['timestamp', 'time', 'TimeStamp', 'datetime', 't', 'ts', 'date']:
            if col in metric_df.columns:
                time_col = col
                break
        if time_col is None:
            # 尝试首列是否为时间
            for col in metric_df.columns:
                if metric_df[col].dtype == 'datetime64[ns]' or (metric_df[col].dtype in ('int64', 'float64') and metric_df[col].min() > 1e9):
                    time_col = col
                    break
        if time_col is not None:
            metric_df['timestamp'] = pd.to_datetime(metric_df[time_col], errors='coerce')
            if time_col != 'timestamp':
                metric_df.drop(columns=[time_col], axis=1, inplace=True)
        else:
            raise ValueError(
                f"Metric数据缺少时间列，期望列: timestamp/time/TimeStamp/datetime，实际列: {list(metric_df.columns)[:20]}..."
            )
        metric_df = metric_df[metric_df['timestamp'].notna()].copy()

        # 统一 service 字段（兼容 TrainTicket 的 PodName；AIOps2025 的 pod / instance）
        if 'service' not in metric_df.columns:
            if 'PodName' not in metric_df.columns:
                if 'pod' in metric_df.columns:
                    metric_df['PodName'] = metric_df['pod'].astype(str)
                elif 'instance' in metric_df.columns:
                    metric_df['PodName'] = metric_df['instance'].astype(str)
                elif 'service_name' in metric_df.columns:
                    metric_df['PodName'] = metric_df['service_name'].astype(str)
                elif 'object_id' in metric_df.columns:
                    metric_df['PodName'] = metric_df['object_id'].astype(str)

            if 'PodName' in metric_df.columns:
                def _pod_to_service(pod: str) -> str:
                    if not isinstance(pod, str):
                        return ''
                    parts = pod.split('-')
                    if len(parts) > 2 and parts[-1].isdigit() and parts[-2].isdigit():
                        return '-'.join(parts[:-2])
                    if len(parts) > 1 and parts[-1].isdigit():
                        return '-'.join(parts[:-1])
                    return pod
                metric_df['service'] = metric_df['PodName'].map(_pod_to_service)
        
        # 过滤时间范围
        if start_time:
            metric_df = metric_df[metric_df['timestamp'] >= start_time]
        if end_time:
            metric_df = metric_df[metric_df['timestamp'] <= end_time]
        
        return metric_df
    
    def aggregate_service_level_metrics(
        self,
        metric_df: pd.DataFrame,
        service_list: List[str],
    ) -> Dict[str, th.Tensor]:
        """
        聚合服务级指标 X_s(t)
        
        特征维度: [P50_Lat, P90_Lat, P99_Lat, QPS, ErrorRate]
        
        Args:
            metric_df: Metric DataFrame
            service_list: 服务列表
        
        Returns:
            {service_name: (T, F) Tensor} T为时间步数，F为特征维度
        """
        print("聚合服务级指标...")
        
        service_metrics = {}
        
        for service_name in tqdm(service_list, desc="处理服务"):
            # 过滤该服务的指标（列不存在时返回全 False 的 mask，避免 KeyError: False）
            mask = pd.Series(False, index=metric_df.index)
            if 'service' in metric_df.columns:
                mask = mask | (metric_df['service'] == service_name)
            if 'service_name' in metric_df.columns:
                mask = mask | (metric_df['service_name'] == service_name)
            service_df = metric_df.loc[mask].copy()
            
            if service_df.empty:
                continue
            
            # 重采样到固定间隔
            service_df.set_index('timestamp', inplace=True)
            service_df_resampled = service_df.resample(f'{self.sampling_interval}s').mean(numeric_only=True)
            
            # 提取特征
            features = []
            
            # P50, P90, P99延迟
            # 优先使用 TrainTicket_2024 的 PodServerLatencyPx(s)，否则回退到 istio/http 字段，否则填 0
            def _get_latency_series(percentile: int) -> np.ndarray:
                # TrainTicket_2024
                if percentile == 50:
                    # 数据里一般没有 P50，退化为 P90（更稳定）
                    for col in ['PodServerLatencyP90(s)', 'PodClientLatencyP90(s)']:
                        if col in service_df_resampled.columns:
                            return service_df_resampled[col].astype(float).fillna(0).values * 1000.0
                if percentile == 90:
                    for col in ['PodServerLatencyP90(s)', 'PodClientLatencyP90(s)']:
                        if col in service_df_resampled.columns:
                            return service_df_resampled[col].astype(float).fillna(0).values * 1000.0
                if percentile == 99:
                    for col in ['PodServerLatencyP99(s)', 'PodClientLatencyP99(s)']:
                        if col in service_df_resampled.columns:
                            return service_df_resampled[col].astype(float).fillna(0).values * 1000.0
                # 通用 Prometheus/istio
                col_name = f'istio_request_duration_milliseconds_p{percentile}'
                if col_name in service_df_resampled.columns:
                    return service_df_resampled[col_name].astype(float).fillna(0).values
                col_name = f'http_request_duration_p{percentile}'
                if col_name in service_df_resampled.columns:
                    return service_df_resampled[col_name].astype(float).fillna(0).values
                for col in ['rrt', 'rrt_max']:
                    if col in service_df_resampled.columns:
                        return service_df_resampled[col].astype(float).fillna(0).values
                return np.zeros(len(service_df_resampled), dtype=float)

            for percentile in [50, 90, 99]:
                features.append(_get_latency_series(percentile))
            
            # QPS/Workload
            if 'PodWorkload(Ops)' in service_df_resampled.columns:
                features.append(service_df_resampled['PodWorkload(Ops)'].astype(float).fillna(0).values)
            else:
                qps_col = 'istio_requests_total'
                if qps_col not in service_df_resampled.columns:
                    qps_col = 'http_requests_total'
                if qps_col not in service_df_resampled.columns:
                    qps_col = 'request'
                if qps_col not in service_df_resampled.columns:
                    qps_col = 'response'
                if qps_col in service_df_resampled.columns:
                    features.append(service_df_resampled[qps_col].astype(float).fillna(0).values)
                else:
                    features.append(np.zeros(len(service_df_resampled), dtype=float))
            
            # ErrorRate
            if 'PodSuccessRate(%)' in service_df_resampled.columns:
                succ = service_df_resampled['PodSuccessRate(%)'].astype(float).fillna(0).values
                error_rate = 1.0 - np.clip(succ / 100.0, 0.0, 1.0)
                features.append(error_rate)
            else:
                ratio_col = None
                for col in ['error_ratio', 'client_error_ratio', 'server_error_ratio']:
                    if col in service_df_resampled.columns:
                        ratio_col = col
                        break
                if ratio_col is not None:
                    ratio = service_df_resampled[ratio_col].astype(float).fillna(0).values
                    if np.max(ratio) > 1.0:
                        ratio = ratio / 100.0
                    features.append(np.clip(ratio, 0.0, 1.0))
                else:
                    error_col = 'istio_request_errors'
                    if error_col not in service_df_resampled.columns:
                        error_col = 'http_request_errors'
                    if error_col not in service_df_resampled.columns:
                        error_col = 'error'
                    total_col = 'istio_requests_total'
                    if total_col not in service_df_resampled.columns:
                        total_col = 'http_requests_total'
                    if total_col not in service_df_resampled.columns:
                        total_col = 'request'
                    if total_col not in service_df_resampled.columns:
                        total_col = 'response'
                    if error_col in service_df_resampled.columns and total_col in service_df_resampled.columns:
                        err = service_df_resampled[error_col].astype(float).fillna(0).values
                        tot = service_df_resampled[total_col].astype(float).fillna(0).values
                        features.append(err / (tot + 1e-6))
                    else:
                        features.append(np.zeros(len(service_df_resampled), dtype=float))
            
            # 组合特征 (T, 5)
            feature_matrix = np.stack(features, axis=1).astype(np.float32)
            
            # Z-Score标准化
            feature_mean = np.mean(feature_matrix, axis=0, keepdims=True)
            feature_std = np.std(feature_matrix, axis=0, keepdims=True) + 1e-6
            feature_matrix = (feature_matrix - feature_mean) / feature_std
            
            service_metrics[service_name] = th.from_numpy(feature_matrix)
        
        print(f"完成服务级指标聚合: {len(service_metrics)} 个服务")
        return service_metrics
    
    def aggregate_pod_level_metrics(
        self,
        metric_df: pd.DataFrame,
        pod_to_service: Dict[str, str],
        candidate_services: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, th.Tensor]]:
        """
        聚合Pod级指标 Z_p(t)
        
        特征维度: [CPU_Usage%, Mem_Usage%, Net_In, Net_Out, Disk_IO, Pod_QPS, Pod_Latency]
        
        Args:
            metric_df: Metric DataFrame
            pod_to_service: Pod到服务的映射
            candidate_services: 候选服务列表（如果提供，只处理这些服务的Pod）
        
        Returns:
            {service_name: {pod_name: (T, F) Tensor}}
        """
        print("聚合Pod级指标...")
        
        pod_metrics = defaultdict(dict)
        
        # 过滤候选服务的Pod
        if candidate_services:
            relevant_pods = [
                pod for pod, svc in pod_to_service.items()
                if svc in candidate_services
            ]
        else:
            relevant_pods = list(pod_to_service.keys())
        
        for pod_name in tqdm(relevant_pods, desc="处理Pod"):
            service_name = pod_to_service[pod_name]
            
            # 过滤该Pod的指标
            pod_df = metric_df[
                (metric_df.get('pod', '') == pod_name) |
                (metric_df.get('pod_name', '') == pod_name)
            ].copy()
            
            if pod_df.empty:
                continue
            
            # 重采样
            pod_df.set_index('timestamp', inplace=True)
            pod_df_resampled = pod_df.resample(f'{self.sampling_interval}s').mean()
            
            # 提取特征
            features = []
            
            # CPU使用率
            cpu_col = 'container_cpu_usage_seconds_total'
            if cpu_col in pod_df_resampled.columns:
                cpu_usage = (pod_df_resampled[cpu_col] * 100).values  # 转换为百分比
                features.append(cpu_usage)
            else:
                features.append(np.zeros(len(pod_df_resampled)))
            
            # 内存使用率
            mem_col = 'container_memory_usage_bytes'
            if mem_col in pod_df_resampled.columns:
                mem_usage = pod_df_resampled[mem_col].values / (1024 * 1024)  # 转换为MB
                features.append(mem_usage)
            else:
                features.append(np.zeros(len(pod_df_resampled)))
            
            # 网络入流量
            net_in_col = 'container_network_receive_bytes_total'
            if net_in_col in pod_df_resampled.columns:
                features.append(pod_df_resampled[net_in_col].values)
            else:
                features.append(np.zeros(len(pod_df_resampled)))
            
            # 网络出流量
            net_out_col = 'container_network_transmit_bytes_total'
            if net_out_col in pod_df_resampled.columns:
                features.append(pod_df_resampled[net_out_col].values)
            else:
                features.append(np.zeros(len(pod_df_resampled)))
            
            # 磁盘IO（简化：使用读取和写入之和）
            disk_read_col = 'container_fs_reads_bytes_total'
            disk_write_col = 'container_fs_writes_bytes_total'
            if disk_read_col in pod_df_resampled.columns and disk_write_col in pod_df_resampled.columns:
                disk_io = (pod_df_resampled[disk_read_col] + 
                          pod_df_resampled[disk_write_col]).values
                features.append(disk_io)
            else:
                features.append(np.zeros(len(pod_df_resampled)))
            
            # Pod QPS（应用级）
            pod_qps_col = 'pod_http_requests_total'
            if pod_qps_col in pod_df_resampled.columns:
                features.append(pod_df_resampled[pod_qps_col].values)
            else:
                features.append(np.zeros(len(pod_df_resampled)))
            
            # Pod延迟
            pod_lat_col = 'pod_http_request_duration_seconds'
            if pod_lat_col in pod_df_resampled.columns:
                features.append(pod_df_resampled[pod_lat_col].values * 1000)  # 转换为毫秒
            else:
                features.append(np.zeros(len(pod_df_resampled)))
            
            # 组合特征 (T, 7)
            feature_matrix = np.stack(features, axis=1).astype(np.float32)
            
            pod_metrics[service_name][pod_name] = th.from_numpy(feature_matrix)
        
        print(f"完成Pod级指标聚合: {len(pod_metrics)} 个服务, "
              f"{sum(len(pods) for pods in pod_metrics.values())} 个Pod")
        
        return dict(pod_metrics)
    
    def compute_normal_baseline(
        self,
        pod_metrics: Dict[str, Dict[str, th.Tensor]],
        baseline_window: int = 30,  # 基线窗口（分钟）
    ) -> Dict[str, th.Tensor]:
        """
        计算正常基线 Z_p^norm（过去30分钟的均值）
        
        Args:
            pod_metrics: Pod指标
            baseline_window: 基线窗口（分钟）
        
        Returns:
            {service_name: baseline_tensor}
        """
        print("计算正常基线...")
        
        baselines = {}
        baseline_steps = baseline_window * 60 // self.sampling_interval
        
        for service_name, pods in pod_metrics.items():
            # 聚合所有Pod的基线
            service_baselines = []
            
            for pod_name, pod_feat in pods.items():
                # 取前baseline_steps个时间步的平均值
                if pod_feat.shape[0] >= baseline_steps:
                    baseline = th.mean(pod_feat[:baseline_steps], dim=0)
                else:
                    baseline = th.mean(pod_feat, dim=0)
                service_baselines.append(baseline)
            
            if service_baselines:
                # 取所有Pod的平均作为服务基线
                baselines[service_name] = th.stack(service_baselines).mean(dim=0)
        
        print(f"完成基线计算: {len(baselines)} 个服务")
        return baselines


def build_service_timeseries(
    trace_parser: TraceParser,
    metric_aggregator: MetricAggregator,
    trace_df: pd.DataFrame,
    call_counts: Dict[Tuple[str, str], int],
    metric_df: pd.DataFrame,
    tau_max: int = 10,
    time_window: int = 60,
) -> Tuple[th.Tensor, List[str], Dict[str, th.Tensor]]:
    """
    构建服务级多变量时间序列 X_s(t)，用于时滞感知双层因果RCA的Stage 1。
    
    整合 Trace 和 Metric 数据，生成统一的服务级时间序列：
    - 从 Trace 提取：请求数、错误率、平均延迟等
    - 从 Metric 提取：CPU、内存、网络等资源指标
    
    Args:
        trace_parser: TraceParser 实例
        metric_aggregator: MetricAggregator 实例
        trace_df: Trace DataFrame（包含 timestamp, parent_service, child_service, duration 等）
        call_counts: 服务调用统计 {(parent_service, child_service): count}
        metric_df: Metric DataFrame
        tau_max: 最大时滞（用于后续历史窗口编码）
        time_window: 时间窗口（秒），用于重采样
    
    Returns:
        (X_tensor, service_list, X_norm):
        - X_tensor: (T, N, F) 张量，T为时间步数，N为服务数，F为特征维度
        - service_list: 服务列表，与 X_tensor 的第1维对应
        - X_norm: {service_name: (F,) Tensor} 正常基线特征（用于差分编码）
    """
    print("=" * 60)
    print("构建服务级时间序列 X_s(t)")
    print("=" * 60)
    
    # Step 1: 获取服务列表和边列表（用于约束图）
    service_list, edge_list = trace_parser.get_trace_topology(call_counts)
    print(f"服务数量: {len(service_list)}")
    print(f"边数量: {len(edge_list)}")
    
    if not service_list:
        raise ValueError("服务列表为空，无法构建时间序列")
    
    # Step 2: 从 Trace 数据构建服务级特征（按时间窗口聚合）
    print("\n[步骤1] 从 Trace 数据提取服务级特征...")
    
    # 确保 trace_df 有时间窗口列
    if 'time_window' not in trace_df.columns:
        trace_df['time_window'] = trace_df['timestamp'].dt.floor(f'{time_window}s')
    
    # 确保 time_window 列是 datetime 类型（过滤掉无效值）
    trace_df = trace_df[trace_df['time_window'].notna()].copy()
    trace_df['time_window'] = pd.to_datetime(trace_df['time_window'], errors='coerce')
    trace_df = trace_df[trace_df['time_window'].notna()].copy()
    
    # 为每个服务在每个时间窗口计算特征
    trace_features = {}
    for service_name in tqdm(service_list, desc="处理服务 Trace 特征"):
        # 过滤该服务的 span（作为 child_service）
        service_traces = trace_df[trace_df['child_service'] == service_name].copy()
        
        if service_traces.empty:
            # 如果该服务没有作为 child 的 trace，尝试作为 parent
            service_traces = trace_df[trace_df['parent_service'] == service_name].copy()
        
        if service_traces.empty:
            continue
        
        # 按时间窗口聚合
        window_stats = service_traces.groupby('time_window').agg({
            'duration': ['count', 'mean', 'std'],  # 请求数、平均延迟、延迟标准差
        }).reset_index()
        
        window_stats.columns = ['time_window', 'request_count', 'avg_latency', 'latency_std']
        window_stats['error_count'] = 0  # 简化：假设 duration > 阈值视为错误
        window_stats['error_rate'] = 0.0
        
        # 计算错误率（简化：duration 异常高视为错误）
        if 'avg_latency' in window_stats.columns:
            latency_threshold = window_stats['avg_latency'].quantile(0.95)
            error_mask = service_traces.groupby('time_window')['duration'].apply(
                lambda x: (x > latency_threshold).sum()
            )
            window_stats['error_count'] = error_mask.values
            window_stats['error_rate'] = error_mask.values / (window_stats['request_count'] + 1e-6)
        
        trace_features[service_name] = window_stats.set_index('time_window')
    
    # Step 3: 从 Metric 数据构建服务级特征
    print("\n[步骤2] 从 Metric 数据提取服务级特征...")
    
    # 使用 MetricAggregator 的现有方法
    metric_service_features = metric_aggregator.aggregate_service_level_metrics(
        metric_df, service_list
    )
    
    # Step 4: 对齐时间轴并合并特征
    print("\n[步骤3] 对齐时间轴并合并特征...")
    
    # 收集所有时间窗口
    all_time_windows = set()
    for df in trace_features.values():
        all_time_windows.update(df.index)
    
    # 从 metric 特征中提取时间轴
    for service_name, metric_tensor in metric_service_features.items():
        # metric_tensor 是 (T, F) 形状，但我们需要知道对应的时间戳
        # 这里简化：假设 metric_df 已经按时间排序，且采样间隔固定
        pass  # 暂时跳过，后续可以从 metric_df 中提取
    
    # 统一时间轴：使用所有 trace 和 metric 的时间窗口的并集
    if all_time_windows:
        # 过滤并转换所有时间窗口为 datetime 对象
        time_axis = []
        for t in all_time_windows:
            # 跳过非 datetime 对象（如 float、int、str 等）
            if isinstance(t, (int, float, str)):
                continue
            # 转换为 Timestamp 然后转为 naive datetime
            try:
                t_ts = pd.Timestamp(t)
                if t_ts.tz is not None:
                    t_ts = t_ts.tz_localize(None)
                time_axis.append(t_ts.to_pydatetime())
            except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
                continue
        time_axis = sorted(time_axis)
    else:
        # 如果没有 trace 特征，从 metric_df 提取时间轴
        if 'timestamp' in metric_df.columns:
            metric_df_sorted = metric_df.sort_values('timestamp')
            time_start = metric_df_sorted['timestamp'].min()
            time_end = metric_df_sorted['timestamp'].max()
            # 统一时区处理
            if pd.Timestamp(time_start).tz is not None:
                time_start = pd.Timestamp(time_start).tz_localize(None)
            if pd.Timestamp(time_end).tz is not None:
                time_end = pd.Timestamp(time_end).tz_localize(None)
            time_axis = pd.date_range(
                start=time_start,
                end=time_end,
                freq=f'{time_window}s'
            ).tolist()
            time_axis = [t.to_pydatetime() if isinstance(t, pd.Timestamp) else t for t in time_axis]
        else:
            raise ValueError("无法确定时间轴")
    
    print(f"时间轴长度: {len(time_axis)} 个时间窗口")
    
    # Step 5: 构建统一特征矩阵 (T, N, F)
    # 特征维度 F = [trace_features: request_count, avg_latency, latency_std, error_rate] 
    #              + [metric_features: P50_Lat, P90_Lat, P99_Lat, QPS, ErrorRate]
    # 总共约 9 维
    
    num_services = len(service_list)
    num_time_steps = len(time_axis)
    num_trace_feats = 4  # request_count, avg_latency, latency_std, error_rate
    num_metric_feats = 5  # P50, P90, P99, QPS, ErrorRate
    num_total_feats = num_trace_feats + num_metric_feats
    
    # 初始化特征矩阵
    X_tensor = th.zeros((num_time_steps, num_services, num_total_feats), dtype=th.float32)
    
    # 填充 trace 特征
    service_to_idx = {svc: i for i, svc in enumerate(service_list)}
    
    for service_name, trace_df_windowed in trace_features.items():
        if service_name not in service_to_idx:
            continue
        
        service_idx = service_to_idx[service_name]
        
        trace_index_map = {}
        for idx in trace_df_windowed.index:
            idx_naive = idx
            if isinstance(idx, pd.Timestamp):
                idx_naive = idx.to_pydatetime()
            if pd.Timestamp(idx_naive).tz is not None:
                idx_naive = pd.Timestamp(idx_naive).tz_localize(None).to_pydatetime()
            trace_index_map[idx_naive] = idx
        
        for t_idx, t_window in enumerate(time_axis):
            # 确保 t_window 是 naive datetime
            t_window_naive = t_window
            if isinstance(t_window, pd.Timestamp):
                t_window_naive = t_window.to_pydatetime()
            if pd.Timestamp(t_window_naive).tz is not None:
                t_window_naive = pd.Timestamp(t_window_naive).tz_localize(None).to_pydatetime()
            
            # 查找匹配的时间窗口
            if t_window_naive in trace_index_map:
                original_idx = trace_index_map[t_window_naive]
                row = trace_df_windowed.loc[original_idx]
                X_tensor[t_idx, service_idx, 0] = row.get('request_count', 0.0)
                X_tensor[t_idx, service_idx, 1] = row.get('avg_latency', 0.0)
                X_tensor[t_idx, service_idx, 2] = row.get('latency_std', 0.0)
                X_tensor[t_idx, service_idx, 3] = row.get('error_rate', 0.0)
       
    # 从 metric_df 构建时间轴（用于对齐）
    if 'timestamp' in metric_df.columns:
        metric_df_sorted = metric_df.sort_values('timestamp')
        metric_time_start = metric_df_sorted['timestamp'].min()
        metric_time_end = metric_df_sorted['timestamp'].max()
        
        # 统一时区处理：转换为 naive datetime
        if pd.Timestamp(metric_time_start).tz is not None:
            metric_time_start = pd.Timestamp(metric_time_start).tz_localize(None)
        if pd.Timestamp(metric_time_end).tz is not None:
            metric_time_end = pd.Timestamp(metric_time_end).tz_localize(None)
        
        # 构建 metric 的时间轴（按 sampling_interval 重采样后的时间点）
        metric_time_axis = pd.date_range(
            start=metric_time_start,
            end=metric_time_end,
            freq=f'{metric_aggregator.sampling_interval}s'
        ).tolist()
        # 转换为 Python datetime 对象（确保时区一致）
        metric_time_axis = [t.to_pydatetime() if isinstance(t, pd.Timestamp) else t for t in metric_time_axis]
    else:
        metric_time_axis = []
    
    for service_name, metric_tensor in metric_service_features.items():
        if service_name not in service_to_idx:
            continue
        
        service_idx = service_to_idx[service_name]
        metric_T, metric_F = metric_tensor.shape
        
        # 对齐 metric 时间轴到统一的 time_axis
        if metric_time_axis:
            # 找到 metric_time_axis 中每个时间点在 time_axis 中的最近索引
            for m_idx, metric_time in enumerate(metric_time_axis[:metric_T]):
                # 确保 metric_time 是 naive datetime
                if isinstance(metric_time, pd.Timestamp):
                    metric_time = metric_time.to_pydatetime()
                if pd.Timestamp(metric_time).tz is not None:
                    metric_time = pd.Timestamp(metric_time).tz_localize(None).to_pydatetime()
                
                # 找到 time_axis 中最接近的时间点
                # 确保 time_axis 中的时间也是 naive datetime
                time_diffs = []
                for t in time_axis:
                    t_naive = t
                    if isinstance(t, pd.Timestamp):
                        t_naive = t.to_pydatetime()
                    if pd.Timestamp(t_naive).tz is not None:
                        t_naive = pd.Timestamp(t_naive).tz_localize(None).to_pydatetime()
                    time_diffs.append(abs((t_naive - metric_time).total_seconds()))
                
                if time_diffs:
                    closest_idx = min(range(len(time_diffs)), key=lambda i: time_diffs[i])
                    # 如果时间差小于 time_window，则填充
                    if time_diffs[closest_idx] < time_window:
                        X_tensor[closest_idx, service_idx, num_trace_feats:num_trace_feats + metric_F] = \
                            metric_tensor[m_idx, :]
        else:
            # 如果没有 metric 时间轴，简化对齐：假设 metric 的时间步与 time_axis 的前 metric_T 个对应
            for t_idx in range(min(metric_T, num_time_steps)):
                X_tensor[t_idx, service_idx, num_trace_feats:num_trace_feats + metric_F] = \
                    metric_tensor[t_idx, :]
    
    # Step 6: 计算正常基线 X_norm（用于差分编码）
    print("\n[步骤4] 计算正常基线 X_norm...")
    
    # 使用前 30% 的时间步作为正常基线（或前 baseline_window 分钟）
    baseline_window_minutes = 30
    baseline_steps = min(
        baseline_window_minutes * 60 // time_window,
        num_time_steps // 3  # 至少使用前 1/3
    )
    
    X_norm = {}
    for service_idx, service_name in enumerate(service_list):
        # 计算前 baseline_steps 个时间步的均值
        baseline_feat = th.mean(X_tensor[:baseline_steps, service_idx, :], dim=0)  # (F,)
        X_norm[service_name] = baseline_feat
    
    print(f"完成服务级时间序列构建:")
    print(f"  - 形状: {X_tensor.shape} (T={num_time_steps}, N={num_services}, F={num_total_feats})")
    print(f"  - 基线窗口: {baseline_steps} 个时间步")
    print(f"  - 服务数量: {len(service_list)}")
    
    return X_tensor, service_list, X_norm


class DatasetConstructor:
    """训练样本构建器"""
    
    def __init__(
        self,
        window_length: int = 60,  # 窗口长度（时间步数）
        positive_label_threshold: float = 0.5,  # 正样本阈值
    ):
        """
        Args:
            window_length: 滑动窗口长度
            positive_label_threshold: 正样本阈值
        """
        self.window_length = window_length
        self.positive_label_threshold = positive_label_threshold
    
    def build_samples(
        self,
        service_metrics: Dict[str, th.Tensor],
        fault_labels: Optional[Dict[datetime, Dict]] = None,
    ) -> Tuple[List[th.Tensor], List[int], List[Dict]]:
        """
        构建训练样本（滑动窗口）
        
        Args:
            service_metrics: 服务级指标 {service_name: (T, F)}
            fault_labels: 故障标签 {timestamp: {service: label, ...}}
        
        Returns:
            (samples, labels, metadata)
            - samples: List[(N_services, T, F)]
            - labels: List[int] (0: 正常, 1: 异常)
            - metadata: List[Dict] 样本元数据
        """
        print("构建训练样本...")
        
        # 确定统一的时间序列长度
        all_lengths = [feat.shape[0] for feat in service_metrics.values()]
        if not all_lengths:
            return [], [], []
        
        min_length = min(all_lengths)
        num_services = len(service_metrics)
        num_features = next(iter(service_metrics.values())).shape[1]
        
        # 对齐所有服务的时间序列（截断到最小长度）
        aligned_metrics = {}
        service_list = sorted(service_metrics.keys())
        
        for service_name in service_list:
            aligned_metrics[service_name] = service_metrics[service_name][:min_length]
        
        # 构建滑动窗口样本
        samples = []
        labels = []
        metadata = []
        
        for i in range(self.window_length, min_length):
            # 提取窗口数据 (N_services, T, F)
            window_data = []
            for service_name in service_list:
                window_feat = aligned_metrics[service_name][i - self.window_length:i]  # (T, F)
                window_data.append(window_feat)
            
            sample = th.stack(window_data, dim=0)  # (N_services, T, F)
            samples.append(sample)
            
            # 确定标签（检查当前时刻是否有故障）
            sample_time = i  # 简化：使用索引
            label = 0
            
            if fault_labels:
                # 检查当前时刻是否在故障时间范围内
                for fault_time, fault_info in fault_labels.items():
                    # 简化：假设fault_time是时间步索引
                    if isinstance(fault_time, datetime):
                        # 需要根据实际时间戳转换
                        pass
                    elif isinstance(fault_time, int):
                        if fault_time <= sample_time <= fault_time + 15:  # 假设故障持续15个时间步
                            label = 1
                            break
            
            labels.append(label)
            
            # 元数据
            metadata.append({
                'time_step': i,
                'services': service_list,
                'window_start': i - self.window_length,
                'window_end': i,
            })
        
        print(f"构建完成: {len(samples)} 个样本, "
              f"正样本: {sum(labels)}, 负样本: {len(labels) - sum(labels)}")
        
        return samples, labels, metadata
    
    def balance_samples(
        self,
        samples: List[th.Tensor],
        labels: List[int],
        metadata: List[Dict],
        negative_ratio: float = 1.0,  # 负样本采样比例
    ) -> Tuple[List[th.Tensor], List[int], List[Dict]]:
        """
        平衡正负样本
        
        Args:
            samples: 样本列表
            labels: 标签列表
            metadata: 元数据列表
            negative_ratio: 负样本采样比例（0-1）
        
        Returns:
            平衡后的 (samples, labels, metadata)
        """
        positive_indices = [i for i, label in enumerate(labels) if label == 1]
        negative_indices = [i for i, label in enumerate(labels) if label == 0]
        
        # 采样负样本
        if negative_ratio < 1.0:
            n_negative = int(len(negative_indices) * negative_ratio)
            sampled_negative = np.random.choice(
                negative_indices, size=n_negative, replace=False
            ).tolist()
        else:
            sampled_negative = negative_indices
        
        # 合并索引
        selected_indices = sorted(positive_indices + sampled_negative)
        
        balanced_samples = [samples[i] for i in selected_indices]
        balanced_labels = [labels[i] for i in selected_indices]
        balanced_metadata = [metadata[i] for i in selected_indices]
        
        print(f"平衡后: {len(balanced_samples)} 个样本, "
              f"正样本: {sum(balanced_labels)}, 负样本: {len(balanced_labels) - sum(balanced_labels)}")
        
        return balanced_samples, balanced_labels, balanced_metadata


class TrainTicketDataPipeline:
    """TrainTicket数据流水线主类"""
    
    def __init__(
        self,
        trace_dir: str,
        metric_dir: str,
        output_dir: str,
        sampling_interval: int = 10,
        window_length: int = 60,
    ):
        """
        Args:
            trace_dir: Trace数据目录
            metric_dir: Metric数据目录
            output_dir: 输出目录
            sampling_interval: 采样间隔（秒）
            window_length: 窗口长度（时间步数）
        """
        self.trace_dir = trace_dir
        self.metric_dir = metric_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.trace_parser = TraceParser(trace_dir)
        self.metric_aggregator = MetricAggregator(metric_dir, sampling_interval)
        self.dataset_constructor = DatasetConstructor(window_length)
    
    def process(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        fault_labels: Optional[Dict] = None,
    ) -> Dict:
        """
        执行完整的数据处理流程
        
        Returns:
            处理结果字典，包含：
            - service_metrics: 服务级指标
            - pod_metrics: Pod级指标
            - adjacency_matrix: 邻接矩阵
            - service_list: 服务列表
            - samples: 训练样本
            - labels: 样本标签
        """
        print("=" * 60)
        print("TrainTicket数据处理流水线")
        print("=" * 60)
        
        # 1. 解析Trace，构建拓扑
        trace_df, call_counts = self.trace_parser.parse_traces(start_time, end_time)
        adjacency_matrix, service_list = self.trace_parser.build_adjacency_matrix(call_counts)
        
        # 2. 加载并聚合Metric
        metric_df = self.metric_aggregator.load_prometheus_metrics(start_time, end_time)
        service_metrics = self.metric_aggregator.aggregate_service_level_metrics(
            metric_df, service_list
        )
        
        # 3. 构建Pod指标（需要pod_to_service映射）
        # 这里简化：假设从metric_df中提取
        pod_to_service = self._extract_pod_to_service_mapping(metric_df)
        pod_metrics = self.metric_aggregator.aggregate_pod_level_metrics(
            metric_df, pod_to_service
        )
        
        # 4. 计算正常基线
        pod_baselines = self.metric_aggregator.compute_normal_baseline(pod_metrics)
        
        # 5. 构建训练样本
        samples, labels, metadata = self.dataset_constructor.build_samples(
            service_metrics, fault_labels
        )
        
        # 6. 保存结果
        results = {
            'service_metrics': service_metrics,
            'pod_metrics': pod_metrics,
            'pod_baselines': pod_baselines,
            'adjacency_matrix': adjacency_matrix,
            'service_list': service_list,
            'samples': samples,
            'labels': labels,
            'metadata': metadata,
        }
        
        self._save_results(results)
        
        return results
    
    def _extract_pod_to_service_mapping(self, metric_df: pd.DataFrame) -> Dict[str, str]:
        """从Metric数据中提取Pod到服务的映射"""
        pod_to_service = {}
        
        if 'pod' in metric_df.columns and 'service' in metric_df.columns:
            mapping_df = metric_df[['pod', 'service']].drop_duplicates()
            pod_to_service = dict(zip(mapping_df['pod'], mapping_df['service']))
        
        return pod_to_service
    
    def _save_results(self, results: Dict):
        """保存处理结果"""
        # 保存服务列表和邻接矩阵
        np.save(os.path.join(self.output_dir, 'adjacency_matrix.npy'), 
                results['adjacency_matrix'])
        with open(os.path.join(self.output_dir, 'service_list.json'), 'w') as f:
            json.dump(results['service_list'], f, indent=2)
        
        # 保存样本和标签
        with open(os.path.join(self.output_dir, 'samples.pkl'), 'wb') as f:
            pickle.dump(results['samples'], f)
        with open(os.path.join(self.output_dir, 'labels.pkl'), 'wb') as f:
            pickle.dump(results['labels'], f)
        
        print(f"结果已保存到: {self.output_dir}")
