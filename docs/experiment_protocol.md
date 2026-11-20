# Standardized Experimental Protocol for 4 × 2 Benchmark Matrix

This document defines the strict, reproducible protocol executed identically across all 8 experimental runs in the research matrix.

---

## 1. Experimental Matrix Overview

```
                      ┌────────────────────────────────────────┐
                      │             Raw Dataset                │
                      │ (CICIDS2017, 2019, 2023, 2025/Faculty) │
                      └──────────────────┬─────────────────────┘
                                         ▼
                      ┌────────────────────────────────────────┐
                      │    Dataset Adapter & Sanitization      │
                      │ (Col Map, Label Bin, Inf/NaN Handling) │
                      └──────────────────┬─────────────────────┘
                                         ▼
                      ┌────────────────────────────────────────┐
                      │      Deterministic 70/15/15 Split      │
                      │ (Train Normal, Val All, Test All)      │
                      └───────┬──────────────────────┬─────────┘
                              │                      │
                   ┌──────────▼────────┐   ┌─────────▼─────────┐
                   │ Model 1: Baseline │   │  Model 2: KAN     │
                   │ (Seq2Seq+Attn)    │   │ (Seq2Seq+Attn+KAN)│
                   └──────────┬────────┘   └─────────┬─────────┘
                              │                      │
                              └──────────┬───────────┘
                                         ▼
                      ┌────────────────────────────────────────┐
                      │     Validation Threshold Search        │
                      │ (Find θ* maximizing Val F1-Score)      │
                      └──────────────────┬─────────────────────┘
                                         ▼
                      ┌────────────────────────────────────────┐
                      │       Frozen Test Evaluation           │
                      │ (Accuracy, F1, ROC, PR, Latency, etc.) │
                      └──────────────────┬─────────────────────┘
                                         ▼
                      ┌────────────────────────────────────────┐
                      │  Standardized JSON/CSV/Figure Logging  │
                      └────────────────────────────────────────┘
```

---

## 2. Step-by-Step Execution Protocol

### Step 1: Environment & Random Seed Initialization
- Set seeds: `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`.
- Set deterministic flags: `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`.
- Log hardware specification, GPU device name/VRAM, CPU cores, Python version, and dependency versions.

### Step 2: Dataset Loading & Adapter Normalization
- Load raw files using the dataset-specific adapter specified in `configs/research.yaml`.
- Execute schema alignment:
  - Map column names to canonical feature representations.
  - Convert multi-class labels to binary ($0 = \text{Benign}, 1 = \text{Attack}$).
  - Replace infinities (`+inf`, `-inf`) with column maximums or 99.9th percentiles.
  - Impute missing values with column medians.

### Step 3: Partitioning & Scaling
- Split data into:
  - **Train partition:** $70\%$
  - **Validation partition:** $15\%$
  - **Test partition:** $15\%$
- Fit `RobustScaler` **exclusively** on the benign samples of the train partition ($y=0$).
- Transform validation and test partitions using the fitted scaler.

### Step 4: Sequence Generation
- Generate rolling temporal windows of length $W = 30$ and stride $S = 1$ (or $S = 5$ for training).
- Label window $y_{\text{seq}} = 1$ if any packet within the window has label $1$, else $0$.
- Filter training set to **only benign sequences** ($y_{\text{seq}} = 0$).

### Step 5: Model Initialization & Equal Capacity Verification
- Instantiate the target model architecture (`baseline_seq2seq` or `kan_seq2seq`).
- Verify identical encoder dimensions, attention parameters, and decoder dimensions.
- Log total trainable parameters and model size in megabytes.

### Step 6: Training & Validation Monitoring
- Train using Adam optimizer ($\text{lr} = 0.001$).
- Apply Cosine Annealing or ReduceLROnPlateau learning rate scheduler.
- Compute composite reconstruction loss (MSE on numerical features + Cross-Entropy on categorical features).
- Track validation loss at each epoch. Trigger Early Stopping if validation loss fails to improve for $7$ consecutive epochs. Save best checkpoint.

### Step 7: Validation Threshold Optimization (Frozen Threshold Selection)
- Load best model checkpoint.
- Compute sequence reconstruction errors $S_{\text{val}}$ across the validation set.
- Execute linear search across 500 candidate threshold values $\theta \in [\min(S_{\text{val}}), \max(S_{\text{val}})]$ to find:
  $$\theta^* = \arg\max_\theta F_1(S_{\text{val}} \ge \theta, y_{\text{val}})$$
- Freeze $\theta^*$ as the permanent decision boundary for this experiment.

### Step 8: Test Set Evaluation & Anomaly Classification
- Compute sequence reconstruction errors $S_{\text{test}}$ across the hold-out test set.
- Generate binary predictions: $\hat{y}_{\text{test}} = \mathbb{I}(S_{\text{test}} \ge \theta^*)$.
- Compute all required primary and secondary metrics:
  - Confusion matrix: $TP, TN, FP, FN$
  - Primary metrics: Accuracy, Precision, Recall, F1-Score
  - Curve metrics: ROC-AUC, PR-AUC
  - Error rates: FPR, FNR

### Step 9: Latency & Throughput Benchmarking
- Warm up model with 50 forward passes.
- Measure inference time for 1,000 individual sequences (batch size 1).
- Compute:
  - Mean Latency (ms)
  - Median Latency (ms)
  - 95th Percentile Latency (ms)
  - 99th Percentile Latency (ms)
  - Inference Throughput (sequences/sec)

### Step 10: Standardized Output Generation
- Save individual metric outputs:
  - `research_results/individual/{dataset}_{model}_metrics.json`
  - `research_results/individual/{dataset}_{model}_metrics.csv`
- Save individual visual figures:
  - Confusion Matrix plot
  - Sequence error histogram vs. threshold
- Record metadata and execution timestamp in `research_results/logs/experiment_run.log`.

---

## 3. Comparison & Synthesis Protocol

Once all 8 experiments are completed:
1. Aggregate metrics into `research_results/comparisons/comparison_table.csv` and `.json`.
2. Generate combined comparative ROC and PR curves per dataset (Baseline vs. KAN).
3. Generate overall latency vs. F1-score scatter plot.
4. Generate publication-ready LaTeX tables (`research_results/tables/latex_results_table.tex`).
5. Compile comprehensive synthesis in `research_results/final_research_report.md`.
