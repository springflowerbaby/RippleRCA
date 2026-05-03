# RippleRCA

RippleRCA is a standalone framework for root cause analysis in microservice systems.

The framework combines:
- topology-constrained lag prior construction
- lag-aware node representation learning
- dual-channel candidate recall for coarse-grained diagnosis
- lag-calibrated counterfactual inference for fine-grained ranking
- preprocessing and evaluation pipelines for RCAbench and AIOps 2025 datasets

## Repository Layout

```text
RippleRCA/
  README.md
  requirements.txt
  .gitignore
  main.py
  lag_aware_rca.py
  config.py
  data_preprocessor.py
  data_pipeline.py
  dataset.py
  dataset_aiops_2025.py
  dataset_rcabench.py
  evaluator.py
  preprocess_rcabench.py
  pcmci_plus_runner.py
  utils.py
  log.py
  mask.py
```
## Datasets

RippleRCA uses publicly available datasets.

- RCAbench: https://zenodo.org/records/17105974
- AIOps 2025: https://www.aiops.cn/gitlab/aiops-live-benchmark/aiopschallenge2025

Please download the datasets from their official sources and place them under the paths used in the commands below.

## Install

```bash
pip install -r requirements.txt
```

## Data Preparation

### RCAbench

```bash
python preprocess_rcabench.py \
  --rcabench_root RCAbench \
  --output_root RCAbench/rcabench_preprocessed
```

## Run

### RCAbench example

```bash
python main.py \
  --dataset_type rcabench \
  --data_dir RCAbench/rcabench_preprocessed \
  --output_dir outputs/rcabench_run \
  --tau_max 2 \
  --top_k_services 8 \
  --top_k_pods 10 \
  --seed 42
```

### AIOps 2025 example

```bash
python main.py \
  --dataset_type aiops2025 \
  --data_dir output \
  --test_dates 20250606 \
  --output_dir outputs/aiops2025_run
```

## Outputs

Each run usually writes:

- `rca_results.json`
- `evaluation_results.json`

The repository also includes reference benchmark outputs under `output/`.
