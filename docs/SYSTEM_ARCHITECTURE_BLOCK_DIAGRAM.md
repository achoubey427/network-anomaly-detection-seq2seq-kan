# End-to-End System Architecture Block Diagram

**Project:** Real-Time Network Anomaly Detection using Seq2Seq LSTM, Bahdanau Attention and KAN  
**Publication Figure:** `research_results/figures/system_architecture_block_diagram.png`

---

## 1. High-Resolution Architecture Block Diagram

![System Architecture Block Diagram](system_architecture_block_diagram.png)

---

## 2. Architectural Pipeline Breakdown

```mermaid
flowchart LR
    A["1. Flow Ingestion<br/>(CSE-CIC-IDS2018 / CIC-IDS2017 / CIC-DDoS2019)"] --> B["2. Leakage-Free Preprocessing<br/>(RobustScaler on y=0, W=30, Strides 5/1)"]
    B --> C["3. Recurrent LSTM Encoder<br/>(2-Layer LSTM, H=64, Dropout=0.2)"]
    C --> D["4. Bahdanau Additive Attention<br/>(Score = v^T tanh(Wq*q + Uk*k))"]
    
    D --> E1["5A. Baseline Bottleneck<br/>(Linear Dense Projection, L=32)"]
    D --> E2["5B. Proposed KAN Bottleneck<br/>(Trainable B-Splines, G=5, k=3)"]
    
    E1 --> F["6. Autoregressive LSTM Decoder<br/>(Context-conditioned Reconstruction)"]
    E2 --> F
    
    F --> G["7. Decision Boundary Evaluation<br/>(Reconstruction MSE vs Validation θ*)"]
    G --> H{"Verdict"}
    H -->|MSE <= θ*| I["NORMAL (0)"]
    H -->|MSE > θ*| J["ANOMALY (1)"]
```

---

## 3. Core Component Specifications

1. **Flow Ingestion:** Ingests official UNB multi-variate network flow records (78–81 canonical flow attributes).
2. **Leakage-Free Preprocessing:** Fits `RobustScaler` strictly on benign training data ($y=0$). Constructs sliding sequences of length $W=30$.
3. **LSTM Encoder:** 2-layer Recurrent LSTM extracting temporal flow trajectories into hidden representation space ($H=64$).
4. **Bahdanau Additive Attention:** Dynamically weighs relevant sequence timesteps to form context vector $c_i$.
5. **Latent Bottlenecks (Controlled Comparison):**
   - **Baseline:** Fixed linear matrix projection ($174,190$ total parameters).
   - **Proposed KAN:** Learnable B-spline univariate functions on $G=5$ intervals with cubic splines $k=3$ ($210,958$ total parameters, $+21.1\%$).
6. **Autoregressive LSTM Decoder:** Reconstructs the multi-variate input sequence timestep by timestep.
7. **Decision Boundary:** Computes Mean Squared Error (MSE). Compares with frozen validation-derived threshold $\theta^*$.

---

## 4. Deliverable Files
- High-Resolution PNG (300 DPI): [`research_results/figures/system_architecture_block_diagram.png`](file:///c:/Users/ASUS/Documents/antigravity/lucid-noether/research_results/figures/system_architecture_block_diagram.png)
- Scalable Vector Graphic (SVG): [`research_results/figures/system_architecture_block_diagram.svg`](file:///c:/Users/ASUS/Documents/antigravity/lucid-noether/research_results/figures/system_architecture_block_diagram.svg)
- Root Directory PNG: [`system_architecture_block_diagram.png`](file:///c:/Users/ASUS/Documents/antigravity/lucid-noether/system_architecture_block_diagram.png)
