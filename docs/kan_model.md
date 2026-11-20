# Proposed Model Specification: Seq2Seq LSTM Autoencoder with Bahdanau Attention and Trainable KAN Latent Bottleneck

**Model Identifier:** `kan_seq2seq`  
**Role in Benchmark:** Proposed Research Model (Model 2)  
**Research Hypothesis:** *Replacing the standard linear latent projection with a learnable B-spline Kolmogorov-Arnold Network (KAN) bottleneck improves multi-variate non-linear flow manifold approximation and anomaly boundary separation without altering encoder/decoder capacity.*

---

## 1. Motivation for KAN in Industrial Anomaly Detection

Traditional deep autoencoders rely on Multi-Layer Perceptrons (MLPs) whose linear weight matrices $W x + b$ are followed by fixed, uniform activation functions $\sigma(\cdot)$ (e.g. ReLU, SiLU). In complex Industrial Control Systems (ICS) and network flow telemetry:
1. Physical sensor telemetry and high-dimensional network flow metrics exhibit complex, localized continuous non-linearities with sharp state transitions (e.g. valve actuations, packet burst dynamics).
2. Standard MLPs suffer from spectral bias and struggle to model high-frequency localized non-linearities without substantial depth.
3. Kolmogorov-Arnold Networks (KANs) place learnable, localized 1D activation functions on the edges (weights) rather than fixed activations at the nodes, providing enhanced function approximation efficiency and localized plasticity.

---

## 2. KAN Mathematical Formulation

According to the Kolmogorov-Arnold representation theorem, any continuous multi-variate function $f: [0, 1]^n \to \mathbb{R}$ on a bounded domain can be represented as a finite composition of continuous univariate functions:

$$f(x_1, \dots, x_n) = \sum_{q=1}^{2n+1} \Phi_q \left( \sum_{p=1}^n \phi_{q,p}(x_p) \right)$$

In our neural KAN layer implementation, the edge activation function $\phi(x)$ combines a residual base linear activation with a linear combination of B-splines:

$$\phi(x) = w_b \cdot b(x) + w_s \cdot \text{Spline}(x)$$

where:
- $b(x) = \text{SiLU}(x) = \frac{x}{1 + e^{-x}}$ provides standard continuous baseline gradient flow.
- $\text{Spline}(x) = \sum_{k=0}^{G + K - 1} c_k B_k(x)$ is a linear combination of $k$-th order B-spline basis functions.
- $G$ is the grid size (number of intervals over $[-1, 1]$).
- $K$ is the spline order (order 3 cubic B-splines by default).
- $c_k$ are trainable spline coefficient parameters.
- $w_b$ and $w_s$ are learnable scaling weights.

---

## 3. Placement in the Anomaly Detection Architecture

The KAN bottleneck replaces the standard affine projection between the attention context and the decoder state initialization:

```
                      ┌──────────────────────────────────────┐
                      │    Input Sequence X: (B, T, D)       │
                      └──────────────────┬───────────────────┘
                                         ▼
                      ┌──────────────────────────────────────┐
                      │      Multi-Layer LSTM Encoder        │
                      └──────────────────┬───────────────────┘
                                         ▼
                      ┌──────────────────────────────────────┐
                      │    Bahdanau Additive Attention       │
                      │       Context Vector c_T             │
                      └──────────────────┬───────────────────┘
                                         ▼
        ┌───────────────────────────────────────────────────────────────┐
        │        PROPOSED KOLMOGOROV-ARNOLD (KAN) BOTTLENECK            │
        │                                                               │
        │  Encoder KAN: z = KANLinear(in=H, out=L)(c_T)                 │
        │  Decoder KAN: s_0 = KANLinear(in=L, out=H)(z)                 │
        │                                                               │
        │  Spline-based edge activations: phi(x) = w_b*SiLU + w_s*B(x) │
        └───────────────────────────────┬───────────────────────────────┘
                                         ▼
                      ┌──────────────────────────────────────┐
                      │      Autoregressive LSTM Decoder     │
                      └──────────────────┬───────────────────┘
                                         ▼
                      ┌──────────────────────────────────────┐
                      │   Reconstruction Output \hat{X}      │
                      └──────────────────┬───────────────────┘
                                         ▼
                      ┌──────────────────────────────────────┐
                      │     Reconstruction Error & Anomaly   │
                      │      Score S(X) = (1/TD) ||X-\hat{X}||² │
                      └──────────────────────────────────────┘
```

---

## 4. Architectural Comparison: Baseline vs. Proposed KAN

| Component | Model 1 (Baseline) | Model 2 (Proposed KAN) | Controlled Status |
|---|---|---|---|
| **Encoder** | 2-Layer LSTM ($H=64$) | 2-Layer LSTM ($H=64$) | **Identical** |
| **Attention** | Bahdanau Additive | Bahdanau Additive | **Identical** |
| **Encoder Bottleneck** | `nn.Linear(H, L)` | `KANLinear(H, L)` ($G=5, K=3$) | **Research Variable** |
| **Decoder Bottleneck** | `nn.Linear(L, H)` | `KANLinear(L, H)` ($G=5, K=3$) | **Research Variable** |
| **Decoder** | 2-Layer LSTM ($H=64$) | 2-Layer LSTM ($H=64$) | **Identical** |
| **Reconstruction Head**| `nn.Linear(H, D)` | `nn.Linear(H, D)` | **Identical** |
| **Loss Function** | MSE Loss | MSE Loss | **Identical** |
| **Training Policy** | Benign ($y=0$) sequences only | Benign ($y=0$) sequences only | **Identical** |
| **Anomaly Scoring** | Mean Squared Reconstruction Error | Mean Squared Reconstruction Error | **Identical** |
| **Thresholding Method**| Validation $F_1$ search ($\theta^*$) | Validation $F_1$ search ($\theta^*$) | **Identical** |

---

## 5. KAN Hyperparameters

Configured in `configs/research.yaml`:
- `grid_size`: $5$ (5 intervals spanning $[-1.0, 1.0]$).
- `spline_order`: $3$ (Cubic B-splines).
- `base_activation`: `"silu"`.
- `grid_eps`: $0.02$.
- `scale_noise`: $0.1$.
- `scale_base`: $1.0$.
- `scale_spline`: $1.0$.

---

## 6. Computational & Gradient Considerations

- **Gradient Flow:** Backward pass propagates gradients simultaneously into base linear weights $w_b$, spline coefficient tensors $c_k$, and spline scaling factors $w_s$.
- **Inference Complexity:** Evaluating B-splines introduces additional element-wise grid lookups relative to standard matrix multiplies, which is systematically quantified via latency benchmarking (ms/sequence).
- **Reproducibility:** B-spline grid buffers are deterministically initialized and registered as persistent PyTorch module buffers.
