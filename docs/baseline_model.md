# Baseline Model Specification: Sequence-to-Sequence LSTM Autoencoder with Bahdanau Attention

**Model Identifier:** `baseline_seq2seq`  
**Role in Benchmark:** Reference Baseline (Model 1)  
**Task:** Unsupervised Behavioral Sequence Reconstruction and Network Anomaly Detection  

---

## 1. Reference-Paper Architecture (Conceptual Inspiration)

The reference work (Zare et al., 2024; Malhotra et al., 2016) explores sequence-to-sequence autoencoders for network intrusion and physical process telemetry. The conceptual baseline comprises:
- An RNN/LSTM encoder that compresses a window of multi-variate packet/flow observations into dynamic hidden states.
- An attention mechanism that assigns dynamic importance weights to individual past timesteps.
- A latent compression bottleneck.
- An autoregressive RNN/LSTM decoder that reconstructs the original sequence under teacher forcing during training.
- Anomaly scoring derived from the magnitude of the reconstruction error between the true input $X$ and reconstructed $\hat{X}$.

---

## 2. Our Baseline Architecture Implementation

Our baseline implementation is structured as a pure sequence-to-sequence reconstruction model designed for variable-dimension network flow tensors:

```
Input Sequence X: (B, T, D)
        │
        ▼
   Multi-Layer LSTM Encoder
        │
        ├───► Hidden States H = [h_1, h_2, ..., h_T]  (B, T, H_dim)
        │
        ▼
Bahdanau Additive Attention Mechanism
  e_{i,j} = v_a^T tanh(W_a s_{i-1} + U_a h_j)
  alpha_{i,j} = softmax_j(e_{i,j})
  c_i = sum_j (alpha_{i,j} * h_j)                      (B, 1, H_dim)
        │
        ▼
Latent Linear Bottleneck: W_enc · c_T -> z             (B, L_dim)
        │
        ▼
Latent Expansion: W_dec · z -> Initial Decoder State s_0 (B, H_dim)
        │
        ▼
Autoregressive LSTM Decoder (with Annealed Teacher Forcing)
  s_t = LSTM_dec([x_{t-1}, c_t], s_{t-1})
        │
        ▼
Output Linear Projection Head: W_out · s_t -> \hat{x}_t (B, T, D)
        │
        ▼
Reconstruction Error & Sequence Anomaly Score
  e_t = ||x_t - \hat{x}_t||_2^2
  S(X) = (1/T) sum_t e_t
```

---

## 3. Similarities with Reference Work

1. **Recurrent Modeling:** Uses multi-layer LSTMs to capture long-range temporal dependencies across network flow sequences.
2. **Dynamic Attention:** Uses additive Bahdanau attention to align decoder reconstruction steps with pertinent encoder timesteps.
3. **Teacher Forcing:** Employs teacher forcing during training to stabilize sequence generation.
4. **Reconstruction-Based Detection:** Unsupervised training strictly on benign traffic ($y=0$), flagging deviations based on reconstruction error magnitude.

---

## 4. Differences from Reference Work

1. **Continuous & Normalized Feature Interface:** Rather than discrete categorical token embeddings for every field, our model accepts standardized multi-variate continuous canonical network features ($D$ dimensions) scaled via training-partition statistics.
2. **Configurable Latent Bottleneck:** The latent dimensionality $L$ is explicitly parameterized to allow strict capacity matching when comparing with the KAN bottleneck in Model 2.
3. **Strict Zero-Teacher-Forcing Inference:** During validation and test evaluation, teacher forcing is strictly forced to $0.0$ so that the model relies entirely on its internal generative representations.

---

## 5. Rationale for Implementation Decisions

| Decision | Scientific Rationale |
|---|---|
| **Dynamic Feature Dimension $D$** | Allows the exact same model code to train on CICIDS2017 ($D=48$), CICIoT2023 ($D=19$), or any custom schema without code modification. |
| **Bahdanau Additive Attention** | Additive attention provides smoother gradient propagation and superior alignment over dot-product attention when encoder and decoder hidden states are transformed. |
| **Separated Latent Bottleneck** | Provides a clean, isolated interface where the standard linear bottleneck in Model 1 can be replaced by the B-spline KAN bottleneck in Model 2 with zero changes to encoder/decoder code. |
| **MSE Reconstruction Loss** | Standardized, well-behaved loss function for continuous scaled feature distributions with outlier clipping. |

---

## 6. Input / Output Dimensions

- **Input Tensor ($X$):** `(batch_size, sequence_length, feature_count)` e.g. `(64, 30, 48)`
- **Encoder Hidden States ($H$):** `(batch_size, sequence_length, hidden_dim)` e.g. `(64, 30, 64)`
- **Attention Context ($c$):** `(batch_size, 1, hidden_dim)` e.g. `(64, 1, 64)`
- **Attention Weights ($\alpha$):** `(batch_size, sequence_length, 1)` e.g. `(64, 30, 1)`
- **Latent Bottleneck ($z$):** `(batch_size, latent_dim)` e.g. `(64, 32)`
- **Reconstructed Output ($\hat{X}$):** `(batch_size, sequence_length, feature_count)` e.g. `(64, 30, 48)`
- **Anomaly Score ($S$):** `(batch_size,)` &mdash; scalar error per sequence.

---

## 7. Loss Function

The training objective is Mean Squared Error (MSE) over the reconstructed sequence:

$$\mathcal{L}_{\text{MSE}}(X, \hat{X}) = \frac{1}{B \cdot T \cdot D} \sum_{b=1}^B \sum_{t=1}^T \sum_{d=1}^D (x_{b,t,d} - \hat{x}_{b,t,d})^2$$

---

## 8. Attention Mechanism

The attention energy $e_{i,j}$ between decoder query state $s_{i-1}$ and encoder annotation $h_j$ is given by:

$$e_{i,j} = v_a^T \tanh(W_a s_{i-1} + U_a h_j)$$

$$\alpha_{i,j} = \frac{\exp(e_{i,j})}{\sum_{k=1}^T \exp(e_{i,k})}$$

$$c_i = \sum_{j=1}^T \alpha_{i,j} h_j$$

The context vector $c_i$ is concatenated with the decoder input at timestep $i$: $\tilde{x}_i = [x_{i-1} \,;\, c_i]$.

---

## 9. Encoder & Decoder Specifications

- **Encoder:** Configurable PyTorch `nn.LSTM(input_size=D, hidden_size=H, num_layers=N, batch_first=True, dropout=p)`.
- **Decoder:** Autoregressive PyTorch `nn.LSTM(input_size=D + H, hidden_size=H, num_layers=N, batch_first=True, dropout=p)`.
- **Output Projection:** Linear layer `nn.Linear(hidden_size, D)` mapping decoder state directly to reconstructed flow feature space.

---

## 10. Latent Bottleneck Representation

- **Compression:** Linear mapping $W_{\text{enc}} \in \mathbb{R}^{H \times L}$ with non-linear activation $\text{ReLU}(W_{\text{enc}} c_T)$.
- **Expansion:** Linear mapping $W_{\text{dec}} \in \mathbb{R}^{L \times H}$ projecting latent code $z$ to initial decoder recurrent states $(h_0, c_0)$.

---

## 11. Anomaly Scoring Formulation

For a given test sequence $X \in \mathbb{R}^{T \times D}$:
1. Pass sequence through model with `teacher_forcing_ratio = 0.0`.
2. Compute pointwise squared error: $E_{t,d} = (x_{t,d} - \hat{x}_{t,d})^2$.
3. Compute sequence anomaly score:
   $$S(X) = \frac{1}{T \cdot D} \sum_{t=1}^T \sum_{d=1}^D E_{t,d}$$
4. Compare $S(X) \ge \theta^*$ (frozen validation threshold) to classify as normal ($0$) or anomalous ($1$).
