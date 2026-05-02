# RippleRCA

RippleRCA is a standalone root cause analysis codebase.

It combines:

- lag-aware causal discovery
- dual-stage service and pod ranking
- RCAbench preprocessing and evaluation
- AIOps-style dataset loading

## Highlights

- Independent repository layout with no runtime dependency on sibling `causelens/RCA`
- Minimal public release focused on the main pipeline
- Relative-path defaults suitable for public release
- Ready to publish with a minimal `.gitignore` and `requirements.txt`

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
  dataset_trainticket_2024.py
  dataset_aiops_2025.py
  dataset_rcabench.py
  evaluator.py
  preprocess_rcabench.py
  pcmci_plus_runner.py
  utils.py
  log.py
  mask.py
```

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

### AIOps-style example

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
- `rerank_debug.json`

