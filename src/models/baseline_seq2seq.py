"""
Baseline Research Model: Sequence-to-Sequence LSTM Autoencoder with Bahdanau Additive Attention.
Reconstructs multi-variate network flow sequences and computes reconstruction-based anomaly scores.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # type: ignore


if TORCH_AVAILABLE:
    class BahdanauAttention(nn.Module):
        """
        Additive Attention Mechanism (Bahdanau et al.).
        Computes attention energies between decoder query state and encoder annotations.
        score(s_{i-1}, h_j) = v_a^T tanh(W_a s_{i-1} + U_a h_j)
        """

        def __init__(self, hidden_dim: int):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.W_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.U_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.v = nn.Linear(hidden_dim, 1, bias=False)

        def forward(self, query: torch.Tensor, keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # query: (batch_size, 1, hidden_dim)
            # keys: (batch_size, seq_len, hidden_dim)
            # scores: (batch_size, seq_len, 1)
            scores = self.v(torch.tanh(self.W_query(query) + self.U_key(keys)))
            weights = F.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
            context = torch.bmm(weights.transpose(1, 2), keys)  # (batch_size, 1, hidden_dim)
            return context, weights


    class BaselineSeq2SeqAutoencoder(nn.Module):
        """
        Baseline Model 1:
        Input Sequence (B, T, D)
          -> LSTM Encoder (B, T, H)
          -> Bahdanau Additive Attention
          -> Latent Linear Bottleneck (B, L)
          -> Initial Decoder Projection (B, H)
          -> Autoregressive LSTM Decoder (B, T, H)
          -> Linear Output Projection (B, T, D)
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            latent_dim: int = 32,
            num_layers: int = 2,
            dropout: float = 0.2,
            bidirectional: bool = False,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.latent_dim = latent_dim
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.num_directions = 2 if bidirectional else 1
            self.encoder_hidden_total = hidden_dim * self.num_directions

            # 1. Recurrent Encoder
            self.encoder_lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )

            # 2. Additive Attention Mechanism
            self.attention = BahdanauAttention(self.encoder_hidden_total)

            # 3. Latent Representation Bottleneck (Linear)
            self.encoder_bottleneck = nn.Linear(self.encoder_hidden_total, latent_dim)
            self.decoder_bottleneck = nn.Linear(latent_dim, hidden_dim)

            # 4. Recurrent Decoder
            self.decoder_lstm = nn.LSTM(
                input_size=input_dim + self.encoder_hidden_total,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

            # 5. Output Reconstruction Projection Head
            self.reconstruction_head = nn.Linear(hidden_dim, input_dim)

        def forward(
            self,
            x: torch.Tensor,
            teacher_forcing_ratio: float = 0.0,
            return_attention: bool = False,
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            # x: (batch_size, seq_len, input_dim)
            batch_size, seq_len, _ = x.shape
            
            # 1. Encode sequence
            encoder_outputs, (h_n, c_n) = self.encoder_lstm(x)  # (batch, seq, encoder_hidden_total)

            # 2. Latent bottleneck projection
            if self.bidirectional:
                last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1).unsqueeze(1)
            else:
                last_hidden = h_n[-1].unsqueeze(1)  # (batch, 1, encoder_hidden_total)

            summary_context, _ = self.attention(last_hidden, encoder_outputs)
            z = F.relu(self.encoder_bottleneck(summary_context.squeeze(1)))  # (batch, latent_dim)

            # Initialize decoder states
            dec_h0 = F.relu(self.decoder_bottleneck(z)).unsqueeze(0).repeat(self.num_layers, 1, 1)
            dec_c0 = torch.zeros_like(dec_h0)

            # 3. Autoregressive Decode
            decoder_input = torch.zeros(batch_size, 1, self.input_dim, device=x.device)
            outputs = []
            attention_history = []
            h_t, c_t = dec_h0, dec_c0

            for t in range(seq_len):
                query = h_t[-1].unsqueeze(1)
                t_context, t_weights = self.attention(query, encoder_outputs)
                if return_attention:
                    attention_history.append(t_weights)

                rnn_input = torch.cat([decoder_input, t_context], dim=-1)
                out, (h_t, c_t) = self.decoder_lstm(rnn_input, (h_t, c_t))
                recon_t = self.reconstruction_head(out)
                outputs.append(recon_t)

                # Teacher forcing check
                if self.training and np.random.rand() < teacher_forcing_ratio and t < seq_len - 1:
                    decoder_input = x[:, t : t + 1, :]
                else:
                    decoder_input = recon_t

            reconstructed = torch.cat(outputs, dim=1)  # (batch, seq_len, input_dim)

            if return_attention:
                att_tensor = torch.cat(attention_history, dim=-1)  # (batch, seq_len, seq_len)
                return reconstructed, att_tensor
            return reconstructed

        def compute_anomaly_score(
            self,
            x,
            aggregation: str = "mean",
            tail_k: int = 5,
            batch_size: int = 256,
        ) -> np.ndarray:
            """
            Compute reconstruction-based anomaly scores per sequence.
            Teacher forcing is strictly disabled (0.0).
            Supports both PyTorch Tensors and memory-mapped NumPy arrays.
            """
            self.eval()
            scores_list = []
            device = next(self.parameters()).device
            with torch.no_grad():
                n_samples = len(x)
                for start_idx in range(0, n_samples, batch_size):
                    batch_slice = x[start_idx : start_idx + batch_size]
                    if isinstance(batch_slice, np.ndarray):
                        batch_x = torch.from_numpy(np.array(batch_slice, dtype=np.float32, copy=True)).to(device)
                    elif isinstance(batch_slice, torch.Tensor):
                        batch_x = batch_slice.to(device)
                    else:
                        batch_x = torch.tensor(batch_slice, dtype=torch.float32).to(device)

                    recon = self.forward(batch_x, teacher_forcing_ratio=0.0, return_attention=False)
                    # Pointwise squared errors: (batch_size, seq_len)
                    errors = torch.mean((batch_x - recon) ** 2, dim=-1)

                    if aggregation == "mean":
                        b_scores = torch.mean(errors, dim=1)
                    elif aggregation == "max":
                        b_scores = torch.max(errors, dim=1).values
                    elif aggregation == "tail_mean":
                        b_scores = torch.mean(errors[:, -tail_k:], dim=1)
                    else:
                        b_scores = torch.mean(errors, dim=1)

                    scores_list.append(b_scores.cpu().numpy())

                return np.concatenate(scores_list, axis=0) if scores_list else np.array([])

else:
    class BaselineSeq2SeqAutoencoder:  # Fallback stub when torch is importing
        def __init__(self, input_dim: int, **kwargs):
            self.input_dim = input_dim
