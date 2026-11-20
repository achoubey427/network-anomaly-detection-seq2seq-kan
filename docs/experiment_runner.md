# Controlled Experiment Runner & Evaluation Framework Specification

**Module:** `src/evaluation/experiment_runner.py`  
**Execution CLI:** `scripts/run_experiment.py`  
**Comparison Utility:** `src/evaluation/comparator.py`  
**Benchmark Matrix:** 4 Datasets &times; 2 Architectures = 8 Controlled Experiments  

---

## 1. Controlled Experiment Matrix

The framework executes a scientifically controlled benchmark across 8 experimental configurations:

| Experiment ID | Dataset | Model Architecture | Training Data | Threshold Source | Evaluation Target |
|---|---|---|---|---|---|
| **EXP-01** | `CICIDS2017` | `baseline_seq2seq` (LSTM-AE + Attention) | Benign only ($y=0$) | Validation set only | Hold-out Test set |
| **EXP-02** | `CICIDS2017` | `kan_seq2seq` (LSTM-AE + Attn + KAN) | Benign only ($y=0$) | Validation set only | Hold-out Test set |
| **EXP-03** | `CICIDS2019` | `baseline_seq2seq` (LSTM-AE + Attention) | Benign only ($y=0$) | Validation set only | Hold-out Test set |
| **EXP-04** | `CICIDS2019` | `kan_seq2seq` (LSTM-AE + Attn + KAN) | Benign only ($y=0$) | Validation set only | Hold-out Test set |
| **EXP-05** | `CICIoT2023` | `baseline_seq2seq` (LSTM-AE + Attention) | Benign only ($y=0$) | Validation set only | Hold-out Test set |
| **EXP-06** | `CICIoT2023` | `kan_seq2seq` (LSTM-AE + Attn + KAN) | Benign only ($y=0$) | Validation set only | Hold-out Test set |
| **EXP-07** | `CICIDS2025` *(CIC-DGG-2025)* | `baseline_seq2seq` (LSTM-AE + Attention) | Benign only ($y=0$) | Validation set only | Hold-out Test set |
| **EXP-08** | `CICIDS2025` *(CIC-DGG-2025)* | `kan_seq2seq` (LSTM-AE + Attn + KAN) | Benign only ($y=0$) | Validation set only | Hold-out Test set |

---

## 2. Execution Commands

### Individual Experiment Execution:
```bash
# Run Baseline on CICIDS2017
python scripts/run_experiment.py --dataset CICIDS2017 --model baseline_seq2seq

# Run Proposed KAN on CICIDS2017
python scripts/run_experiment.py --dataset CICIDS2017 --model kan_seq2seq
```

### Full Benchmark Execution:
```bash
# Sequentially run all available experiments in the 4x2 matrix
python scripts/run_experiment.py --all
```

---

## 3. Standardized Experiment Workflow

```
1. Dataset Availability & Integrity Check
   ├── If missing -> Log status: DATASET_MISSING, print missing path, terminate cleanly.
   └── If present -> Proceed to Step 2.
           │
           ▼
2. Load Partitioned Sequence Tensors (data/processed/{dataset}/)
   ├── X_train: (N_train, 30, D)  [Benign sequences only, y = 0]
   ├── X_val:   (N_val, 30, D)    [Mixed normal and anomalous sequences]
   └── X_test:  (N_test, 30, D)   [Held-out mixed test set]
           │
           ▼
3. Instantiate Target Architecture (build_model: baseline_seq2seq OR kan_seq2seq)
           │
           ▼
4. Train Autoencoder (Adam, lr=0.001, early stopping patience=7 on Val MSE)
   ├── Checkpoint best model on validation loss improvement
   └── Save training history log
           │
           ▼
5. Validation Threshold Optimization (Frozen Threshold Selection)
   ├── Compute validation reconstruction errors S_val with teacher_forcing=0.0
   ├── Linear grid search over 500 candidate thresholds theta
   ├── Select theta* = argmax_theta F1(S_val >= theta, y_val)
   └── Save threshold.json (theta*, val_f1, val_precision, val_recall)
           │
           ▼
6. Frozen Test Set Evaluation (Executed exactly ONCE)
   ├── Compute test reconstruction errors S_test with teacher_forcing=0.0
   ├── Apply frozen decision boundary: y_pred = (S_test >= theta*)
   └── Compute complete metric suite (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, FPR, FNR, Latency)
           │
           ▼
7. Persist Deterministic Experiment Artifacts
   └── research_results/experiments/{dataset}/{model}/
```

---

## 4. Threshold Selection & Isolation Protocol

> [!IMPORTANT]
> The test partition is strictly isolated and never accessed during model training, hyperparameter tuning, scaler fitting, or threshold selection.

- **Objective:** $\theta^* = \arg\max_{\theta} F_1(\hat{y}_{\text{val}}(\theta), y_{\text{val}})$.
- **Search Grid:** 500 uniformly spaced intervals between $\min(S_{\text{val}})$ and $\max(S_{\text{val}})$.
- **Fallback Policy:** If validation set contains zero attack samples (pure unsupervised evaluation), the threshold defaults to the empirical 99th percentile: $\theta^* = \text{Percentile}(S_{\text{val}}, 99.0)$.
- **Freezing:** Once $\theta^*$ is selected, it is saved into `threshold.json` and applied unchanged to the test set.

---

## 5. Experiment Artifact Structure

All artifacts are persisted into a deterministic, isolated folder hierarchy:

```
research_results/experiments/{dataset}/{model}/
├── config.json                 # Complete snapshot of experimental parameters
├── training_history.json       # Epoch-by-epoch train/val loss, LR, and elapsed times
├── metrics.json                # Final test metrics (Accuracy, F1, ROC-AUC, Latency, etc.)
├── threshold.json              # Optimized validation threshold theta* and search metadata
├── predictions.npz             # Test ground truth, reconstruction scores, and binary predictions
├── experiment_metadata.json    # Hardware, seed, PyTorch version, parameters, timestamp
└── checkpoint.pt               # PyTorch model state_dict and optimizer weights
```

---

## 6. Reproducibility & Integrity Controls

- **Fixed Seed:** $42$ configured for Python, NumPy, and PyTorch autograd.
- **Status Lifecycle:** Every experiment run transitions through deterministic states:
  - `DATASET_MISSING`: Dataset files not present in `data/raw/`; run skipped cleanly without generating fake metrics.
  - `RUNNING`: Experiment currently executing.
  - `COMPLETED`: Experiment successfully finished on real data; all artifacts persisted.
  - `FAILED`: Experiment encountered an execution exception; traceback recorded in log.
- **Zero Fabrication Policy:** Placeholder metrics (e.g. `0.0` or synthetic scores) are never written into research tables for missing datasets.

---

## 7. Comparative Analysis Framework

The comparator utility (`src/evaluation/comparator.py`) aggregates completed experiments and generates:
1. `research_results/comparisons/baseline_vs_kan_{dataset}.json`
2. `research_results/comparisons/overall_comparison_table.csv`
3. `research_results/comparisons/overall_comparison_table.json`
4. Publication-ready LaTeX tables (`research_results/tables/comparison_table.tex`).
