# Research Upgrade Plan: Controlled Benchmark of Seq2Seq LSTM-Attention vs. KAN-Enhanced Autoencoder for ICS Anomaly Detection

**Project Title:** Real-Time Network-Based Anomaly Detection in Industrial Control Systems  
**Subject:** Intelligent Model Design (IMD) Research Upgrade  
**Target Matrix:** 4 Datasets × 2 Architectures = 8 Controlled Experiments  

---

## A. Research Objective

The primary objective of this research is to rigorously evaluate and compare two deep generative sequence architectures for behavioral anomaly detection in Industrial Control System (ICS) / SCADA network traffic:
1. **Baseline Model (Model 1):** Sequence-to-Sequence (Seq2Seq) LSTM Autoencoder with Bahdanau Additive Attention.
2. **Proposed Model (Model 2):** Sequence-to-Sequence (Seq2Seq) LSTM Autoencoder with Bahdanau Attention and a Kolmogorov-Arnold Network (KAN) non-linear latent bottleneck.

The core research question asks:
> *Does integrating a parameterized B-spline Kolmogorov-Arnold Network (KAN) bottleneck into a Seq2Seq LSTM-Attention Autoencoder improve temporal pattern reconstruction, anomaly discrimination, and boundary separation across diverse benchmark network intrusion datasets without causing prohibitive inference latency?*

---

## B. Research Problem

Industrial Control Systems (ICS) and SCADA environments rely heavily on legacy fieldbus and Ethernet protocols (e.g., Modbus/TCP, DNP3, EtherNet/IP) that lack intrinsic encryption, authentication, or tamper verification. Consequently, stealthy cyber-attacks (such as false data injection, register manipulation, and unauthorized function code execution) can operate within protocol-valid syntactic boundaries while causing catastrophic physical anomalies.

Traditional signature-based Intrusion Detection Systems (IDS) fail against zero-day exploits and subtle behavioral deviations. Unsupervised reconstruction-based autoencoders learn normal behavioral dynamics and flag unexpected deviations. However, standard Multi-Layer Perceptrons (MLPs) and dense latent layers use fixed activation functions on linear node combinations, which can suffer from:
- Inefficient representation of complex non-linear temporal correlations in high-dimensional feature spaces.
- Blurry reconstruction of sharp transition points in physical sensor and packet telemetry.
- Susceptibility to catastrophic forgetting when traffic dynamics drift.

Kolmogorov-Arnold Networks (KANs) replace fixed node activations with learnable, univariate B-spline activation functions on network edges, offering enhanced function approximation power, superior interpretability, and potential parameter efficiency for complex multi-variate continuous physical system interactions.

---

## C. Baseline Model Architecture (Model 1)

```
Input Sequence [B, T, D_in]
        │
        ▼
   LSTM Encoder (Bidirectional/Unidirectional Multi-layer)
        │
        ├───► Hidden States H = [h_1, h_2, ..., h_T]
        │
        ▼
Bahdanau Additive Attention Mechanism
  score(s_{i-1}, h_j) = v_a^T tanh(W_a s_{i-1} + U_a h_j)
  Context Vector c_i = sum(alpha_{ij} * h_j)
        │
        ▼
   Latent Representation z (Dense Linear Projection)
        │
        ▼
   LSTM Decoder (Teacher Forcing Annealing during Training)
        │
        ▼
Dual Reconstruction Heads:
  - Continuous/Numerical Features: Regression Head (MSE Loss)
  - Discrete/Categorical Features: Classification Head (Cross-Entropy Loss)
        │
        ▼
Reconstruction Error e_t = ||x_t - x^_t||_2
        │
        ▼
Anomaly Score S(X) = Aggregate(e_1, ..., e_T)
```

- **Encoder:** Multi-layer LSTM capturing sequential flow temporal dependencies across window size $T$.
- **Attention:** Bahdanau additive attention focusing dynamically on critical transient states.
- **Latent Bottleneck:** Standard dense linear mapping $\mathbb{R}^{H} \to \mathbb{R}^{L} \to \mathbb{R}^{H}$.
- **Decoder:** Autoregressive LSTM reconstructing the input sequence from context and previous outputs.

---

## D. Proposed Model Architecture (Model 2: KAN-Enhanced)

```
Input Sequence [B, T, D_in]
        │
        ▼
   LSTM Encoder (Identical parameters to Model 1)
        │
        ├───► Hidden States H = [h_1, h_2, ..., h_T]
        │
        ▼
Bahdanau Additive Attention Mechanism
        │
        ▼
   Latent Representation z_raw
        │
        ▼
┌────────────────────────────────────────────────────────┐
│ Kolmogorov-Arnold Network (KAN) Latent Bottleneck      │
│                                                        │
│ Phi(x) = sum_{i=1}^n phi_i(x_i)                        │
│ phi(x) = w_b * b(x) + w_s * Spline(x)                  │
│ where Spline(x) = sum_k c_k * B_k(x) (Order-3 B-spline)│
│                                                        │
│ Configurable: grid_size=5, k=3, grid_range=[-1, 1]     │
└────────────────────────────────────────────────────────┘
        │
        ▼
   Non-linear Latent Feature z_kan
        │
        ▼
   LSTM Decoder (Identical architecture to Model 1)
        │
        ▼
Dual Reconstruction Heads (MSE + Cross-Entropy)
        │
        ▼
Reconstruction Error & Anomaly Score
```

- **KAN Bottleneck:** Replaces the standard affine linear transformation with a learnable B-spline activation network ($z \in \mathbb{R}^{H} \xrightarrow{\text{KAN}} \mathbb{R}^{K} \xrightarrow{\text{KAN}} \mathbb{R}^{H}$).
- **Spline Representation:** Uses a $k$-th order B-spline basis over a dynamic grid updated via moving average activations during training.
- **Equalized Dimension:** Latent capacity is matched with Model 1 to ensure that any performance delta is strictly attributable to the functional form of KAN, not parameter count inflation.

---

## E. Four Benchmark Datasets

| Dataset Identifier | Domain / Source | Key Characteristics | Target Attacks | Schema Strategy |
|---|---|---|---|---|
| **CICIDS2017** | Enterprise / Perimeter Network Flows | 80+ network flow features (Flow Duration, Packet Lengths, Flags, Inter-arrival times). | DoS, DDoS, PortScan, Brute Force, Web Attacks, Infiltration. | Standard CICFlowMeter format; label column `Label`. |
| **CICIDS2019** | Modern Multi-day Attack Flows | High-volume reflection & exploitation traffic (NetBIOS, LDAP, MSSQL, Syn, UDP). | Modern DDoS, SYN Floods, UDP Floods. | CSV columns with whitespace; normalized via adapter mapping. |
| **CICIDS2023** | IoT & Industrial Network Topologies | 47 extracted features across 33 attack classes and benign traffic. | Mirai, DoS/DDoS, Reconnaissance, Spoofing, Brute Force. | IoT/SCADA edge flow features; unified label mapping. |
| **CICIDS2025** *(CIC-DGG-2025)* | Modern Distributed / Generative Threat Framework | Faculty-specified target benchmark dataset. | Advanced persistent threats, zero-day evasion, distributed telemetry. | Flexible dynamic schema detection; adapter interface ready. |

---

## F. 4 × 2 Experiment Matrix

| Exp ID | Dataset | Model Architecture | Training Data | Evaluation Target |
|---|---|---|---|---|
| **EXP-01** | CICIDS2017 | Seq2Seq LSTM-Attention (Baseline) | Normal sequences only | Full Test Set (Normal + Attacks) |
| **EXP-02** | CICIDS2017 | Seq2Seq LSTM-Attention + KAN | Normal sequences only | Full Test Set (Normal + Attacks) |
| **EXP-03** | CICIDS2019 | Seq2Seq LSTM-Attention (Baseline) | Normal sequences only | Full Test Set (Normal + Attacks) |
| **EXP-04** | CICIDS2019 | Seq2Seq LSTM-Attention + KAN | Normal sequences only | Full Test Set (Normal + Attacks) |
| **EXP-05** | CICIDS2023 | Seq2Seq LSTM-Attention (Baseline) | Normal sequences only | Full Test Set (Normal + Attacks) |
| **EXP-06** | CICIDS2023 | Seq2Seq LSTM-Attention + KAN | Normal sequences only | Full Test Set (Normal + Attacks) |
| **EXP-07** | CICIDS2025 | Seq2Seq LSTM-Attention (Baseline) | Normal sequences only | Full Test Set (Normal + Attacks) |
| **EXP-08** | CICIDS2025 | Seq2Seq LSTM-Attention + KAN | Normal sequences only | Full Test Set (Normal + Attacks) |

---

## G. Preprocessing Strategy

1. **Dataset Adapter Layer:** Each dataset has a dedicated adapter class subclassing `BaseDatasetAdapter`.
2. **Column Sanitization:** Strip leading/trailing whitespaces, uniform lowercase transformation, special character replacement.
3. **Missing and Infinite Value Handling:**
   - Detect `NaN`, `null`, `inf`, `-inf`.
   - Replace infinities with maximum non-infinite float32 values or column 99.9th percentile.
   - Impute missing numerical values using median computed strictly on training partition.
4. **Label Binarization:**
   - Normal/Benign/Background $\to 0$.
   - Malicious/Attack/Anomaly $\to 1$.
   - Multi-class string labels mapped deterministically via YAML configuration.
5. **Feature Scaling:**
   - RobustScaler or MinMaxScaler fitted **only** on the normal training split.
   - Outlier clipping applied at $[-5, 5]$ (for standard scaling) or $[0, 1]$ (for MinMax).
6. **Sliding Window Sequence Generation:**
   - Window Length $W = 30$ packets/flows.
   - Stride $S = 1$ (for test streaming) or $S = 5$ (for memory-efficient training).
   - Sequence label: A window is labeled anomalous ($1$) if $\ge 1$ packet within the window is malicious.

---

## H. Training Strategy

- **Unsupervised Anomaly Modeling:** Models are trained strictly on benign ($y=0$) traffic sequences so the network learns the underlying normal manifold.
- **Loss Function:** Composite reconstruction loss:
  $$\mathcal{L} = \frac{1}{T \cdot D_{\text{num}}} \sum_{t=1}^T ||x_t^{\text{num}} - \hat{x}_t^{\text{num}}||_2^2 + \lambda \sum_{t=1}^T \text{CrossEntropy}(x_t^{\text{cat}}, \hat{x}_t^{\text{cat}})$$
- **Optimizer:** Adam with initial learning rate $\eta = 10^{-3}$, weight decay $10^{-5}$.
- **Learning Rate Schedule:** ReduceLROnPlateau (factor $0.5$, patience $3$).
- **Early Stopping:** Monitor validation reconstruction loss (patience $7$ epochs).
- **Teacher Forcing:** Decaying ratio from $1.0 \to 0.2$ over epochs.

---

## I. Threshold Selection Strategy

> [!IMPORTANT]
> The test set is completely held out during threshold calibration to prevent data snooping.

1. **Validation Score Distribution:** Compute sequence reconstruction errors on the validation partition (which contains both normal and validation anomaly sequences).
2. **Optimization Objective:** Find threshold $\theta^*$ that maximizes the F1 score on validation data:
   $$\theta^* = \arg\max_{\theta} F_1(\text{score}_{\text{val}}, y_{\text{val}}, \theta)$$
3. **Statistical Fallback:** If validation set contains zero attacks (pure unsupervised regime), set threshold via empirical percentile:
   $$\theta^* = \mu_{\text{val}} + k \cdot \sigma_{\text{val}} \quad \text{or} \quad \text{Percentile}(\text{score}_{\text{val}}, 99.0)$$
4. **Frozen Application:** Apply fixed $\theta^*$ directly to test partition anomaly scores.

---

## J. Evaluation Metrics

For each experiment, compute and log:
1. **Classification Metrics:**
   - Accuracy: $(TP + TN) / (TP + TN + FP + FN)$
   - Precision: $TP / (TP + FP)$
   - Recall (Sensitivity): $TP / (TP + FN)$
   - F1-Score: $2 \cdot (P \cdot R) / (P + R)$
   - Specificity: $TN / (TN + FP)$
   - False Positive Rate (FPR): $FP / (FP + TN)$
   - False Negative Rate (FNR): $FN / (TP + FN)$
2. **Curve Metrics (Threshold-Independent):**
   - Area Under the ROC Curve (ROC-AUC)
   - Precision-Recall Area Under Curve (PR-AUC)
3. **Confusion Matrix Counts:**
   - $TP, TN, FP, FN$
4. **Operational / Efficiency Metrics:**
   - Single-sequence inference latency (mean, median, p95, p99 in milliseconds)
   - Throughput (sequences/sec and packets/sec)
   - Model parameter count & memory footprint (MB)

---

## K. Expected Experiment Outputs

All outputs are saved to structured, machine-readable, and publication-ready formats:
- `research_results/individual/{dataset}_{model}_metrics.json`
- `research_results/individual/{dataset}_{model}_metrics.csv`
- `research_results/comparisons/comparison_table.csv`
- `research_results/comparisons/comparison_table.json`
- `research_results/figures/roc_curves_{dataset}.png`
- `research_results/figures/pr_curves_{dataset}.png`
- `research_results/figures/confusion_matrix_{dataset}_{model}.png`
- `research_results/figures/latency_comparison.png`
- `research_results/figures/training_loss_curves_{dataset}.png`
- `research_results/tables/latex_results_table.tex`
- `research_results/final_research_report.md`

---

## L. Reproducibility Strategy

1. **Deterministic Random Seeds:** Seed $42$ configured for Python `random`, `numpy`, and PyTorch (`torch.manual_seed`, `torch.cuda.manual_seed_all`, `torch.backends.cudnn.deterministic = True`).
2. **Traceability Logging:** Every run logs:
   - Git commit hash
   - OS & Python version
   - PyTorch / CUDA version
   - CPU / GPU device details
   - Complete YAML configuration snapshot
   - Timestamp and execution duration

---

## M. Real-Time Deployment & Dashboard Strategy

- The real-time detection pipeline remains fully operational.
- The `SequenceBuffer` maintains per-flow packet histories and performs streaming window inference.
- The model loader supports switching between the Baseline model and the KAN-enhanced model dynamically via configuration or API endpoint.
- Alerts generated by either model stream seamlessly over WebSocket to the React dashboard and persist to SQLite.

---

## N. Limitations

1. **Label Noise & Ambiguity:** Public benchmark intrusion datasets may contain mislabeled background packets or unrecorded scan artifacts.
2. **Computational Overhead of KAN:** B-spline evaluation over dynamic grids introduces additional compute overhead during training and backpropagation compared to dense matrix multiplications.
3. **Zero-Day Generalization:** Pure reconstruction methods may exhibit false positives on legitimate but previously unseen traffic spikes (e.g. maintenance cycles).

---

## O. Future Extension Points

1. Integration of graph neural network (GNN) spatial topology representations with temporal KAN sequence models.
2. Multi-protocol ICS parsers extending beyond Modbus/TCP to IEC 60870-5-104, DNP3, and OPC UA.
3. On-device edge deployment with TensorRT / ONNX quantization for low-power industrial RTUs and PLCs.
