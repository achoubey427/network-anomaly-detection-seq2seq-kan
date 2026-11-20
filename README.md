# Network Anomaly Detection: Seq2Seq LSTM + Attention + KAN

An empirical research benchmark and real-time detection platform comparing **Sequence-to-Sequence (Seq2Seq) LSTM Autoencoders with Bahdanau Additive Attention** against a proposed **Kolmogorov-Arnold Network (KAN) Latent Bottleneck Autoencoder** across official Canadian Institute for Cybersecurity (CIC) benchmark datasets.

---

## 🌐 Live Production Demo & Access

- **Public Production Web App:** [https://lucid-noether.vercel.app](https://lucid-noether.vercel.app)
- **Local Run Command:**
  ```bash
  python src/api/server.py --host 0.0.0.0 --port 8000
  ```
- **Local UI Access:** Open `http://localhost:8000` in any browser.

---

## 🔬 Research Overview & Problem Formulation

Network-based anomaly detection in modern enterprise and Industrial Control System (ICS) topologies requires continuous, non-linear modeling of high-dimensional multi-variate network flow sequences. 

### Architecture Comparison
1. **Model 1 (Baseline):** 2-layer Recurrent LSTM Encoder $\rightarrow$ Bahdanau Additive Attention $\rightarrow$ Linear Latent Bottleneck ($L=32$) $\rightarrow$ Autoregressive 2-layer LSTM Decoder ($174,190$ trainable parameters).
2. **Model 2 (Proposed KAN):** 2-layer Recurrent LSTM Encoder $\rightarrow$ Bahdanau Additive Attention $\rightarrow$ **Trainable B-spline Kolmogorov-Arnold Network (KAN) Bottleneck** ($G=5, k=3$) $\rightarrow$ Autoregressive 2-layer LSTM Decoder ($210,958$ trainable parameters).

### Scientific Rigor & Protocols
- **Zero Fabrication:** 100% official UNB/CIC datasets.
- **Strict Data Isolation:** Scalers fit strictly on normal training data ($y=0$); decision thresholds $\theta^*$ derived strictly on validation sets via $F_1$-optimization.
- **Identical Test Partitions:** Exact same test sequences, indices, and features evaluated across both architectures.
- **Time-Bounded Protocol:** CSE-CIC-IDS2018 was trained to full convergence. CIC-IDS2017 KAN was evaluated on the saved Epoch 3 checkpoint under the time-bounded protocol. CIC-DDoS2019 was executed under the approved time-bounded budget (`MAX_EPOCHS=3`, `EARLY_STOPPING_PATIENCE=1`).

---

## 📊 Consolidated Benchmark Results

| Dataset | Model Architecture | Trainable Params | Optimal Threshold ($\theta^*$) | Test Accuracy | Precision | Recall | $F_1$-Score | ROC-AUC | PR-AUC | Latency (ms/seq) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CSE-CIC-IDS2018** | Baseline Seq2Seq | 174,190 | 0.604098 | 0.9999 | 0.9999 | 1.0000 | **0.9999** | 0.5496 | 0.9999 | 23.00 |
| **CSE-CIC-IDS2018** | **Proposed KAN** | 210,958 | 0.156866 | 0.9999 | 0.9999 | 1.0000 | **0.9999** | 0.4994 | 0.9999 | 23.39 |
| **CIC-IDS2017** | Baseline Seq2Seq | 174,190 | 0.619462 | 0.9986 | 0.9986 | 1.0000 | **0.9993** | 0.6519 | 0.9991 | 17.25 |
| **CIC-IDS2017** | **Proposed KAN** | 210,958 | 0.093085 | 0.9986 | 0.9986 | 1.0000 | **0.9993** | **0.7791** | **0.9996** | 27.73 |
| **CIC-DDoS2019** | Baseline Seq2Seq | 175,921 | 3.056587 | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.5000 | 1.0000 | 18.40 |
| **CIC-DDoS2019** | **Proposed KAN** | 212,689 | 3.071186 | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.5000 | 1.0000 | 18.34 |

*Note: CICIoT2023 was not included in the final experimental evaluation.*

---

## 📁 Datasets Evaluated

1. **CSE-CIC-IDS2018**: Official UNB enterprise intrusion dataset (78 features, 49.6k test sequences).
2. **CIC-IDS2017**: Official UNB ISCX benchmark (78 features, 424.5k test sequences).
3. **CIC-DDoS2019**: Official UNB modern reflection DDoS benchmark (81 features, 270k test sequences).

---

## 🚀 Installation & Local Execution

### 1. Requirements
- Python 3.10+
- PyTorch 2.2+ (CPU or CUDA)
- Node.js 18+ (for building frontend)

```bash
git clone https://github.com/abcd1244/lucid-noether.git
cd lucid-noether

# Install Python requirements
pip install -r requirements-local.txt

# (Optional) Build Frontend SPA
cd frontend
npm install
npm run build
cd ..
```

### 2. Start Live Inference Web Application
```bash
python src/api/server.py --port 8000
```
Open `http://localhost:8000` to interact with the live detection demo, switch models, select real held-out test packet flows, and inspect feature error breakdowns.

---

## 🛠️ Demonstration Workflow (Viva Guide)
1. Open the application at [https://lucid-noether.vercel.app](https://lucid-noether.vercel.app) or `http://localhost:8000`.
2. Navigate to **Live Detection**.
3. Select an architecture: **Baseline Seq2Seq** or **Proposed KAN**.
4. Choose a verified held-out flow sample (e.g. *Benign Web HTTP Traffic* or *PortScan Horizontal Probe*).
5. Click **Run Real Detection**.
6. Observe the immediate real-time verdict (**NORMAL** vs. **ANOMALY**), exact reconstruction error, frozen decision threshold ($\theta^*$), measured inference latency (~$18-27\text{ ms}$), and per-feature error breakdown.
7. Switch to the opposite model to compare the representational boundaries and decision sensitivity.

---

## 📑 Research Reports & Documentation
- **Consolidated Research Comparison:** [`docs/FINAL_RESEARCH_COMPARISON.md`](docs/FINAL_RESEARCH_COMPARISON.md)
- **Comprehensive Benchmark Report:** [`research_results/final_research_report.md`](research_results/final_research_report.md)
- **Publication Figures:** [`research_results/figures/`](research_results/figures/)
- **Data Integrity Audit:** [`docs/AUDIT_REPORT.md`](docs/AUDIT_REPORT.md)
