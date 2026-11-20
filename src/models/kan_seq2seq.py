"""
Proposed Research Model (Model 2): Sequence-to-Sequence LSTM Autoencoder with Attention and KAN Bottleneck.
Integrates a Kolmogorov-Arnold Network (KAN) latent bottleneck with learnable B-spline univariate functions.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.kan_layer import KANLinear
from src.models.baseline_seq2seq import BahdanauAttention


class KANSeq2SeqAutoencoder(nn.Module):
    """
    Model 2 (Proposed KAN-Enhanced):
    Input Sequence (B, T, D)
      -> LSTM Encoder (B, T, H)
      -> Bahdanau Additive Attention
      -> Kolmogorov-Arnold Network (KAN) Bottleneck (B, L)
      -> KAN Latent Decoder Projection (B, H)
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
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
        kan_grid_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.encoder_hidden_total = hidden_dim * self.num_directions

        # 1. Recurrent Encoder (Identical to Baseline)
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # 2. Additive Attention Mechanism (Identical to Baseline)
        self.attention = BahdanauAttention(self.encoder_hidden_total)

        # 3. KAN Latent Bottleneck (Kolmogorov-Arnold Network)
        self.kan_encoder_bottleneck = KANLinear(
            in_features=self.encoder_hidden_total,
            out_features=latent_dim,
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
            grid_range=kan_grid_range,
        )
        self.kan_decoder_bottleneck = KANLinear(
            in_features=latent_dim,
            out_features=hidden_dim,
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
            grid_range=kan_grid_range,
        )

        # 4. Recurrent Decoder (Identical to Baseline)
        self.decoder_lstm = nn.LSTM(
            input_size=input_dim + self.encoder_hidden_total,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 5. Output Reconstruction Projection Head (Identical to Baseline)
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
        encoder_outputs, (h_n, c_n) = self.encoder_lstm(x)

        # 2. Latent bottleneck projection via KAN
        if self.bidirectional:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1).unsqueeze(1)
        else:
            last_hidden = h_n[-1].unsqueeze(1)

        summary_context, _ = self.attention(last_hidden, encoder_outputs)
        
        # Non-linear B-spline KAN transformation
        z = self.kan_encoder_bottleneck(summary_context.squeeze(1))  # (batch, latent_dim)
        dec_h0_base = self.kan_decoder_bottleneck(z)                # (batch, hidden_dim)

        dec_h0 = dec_h0_base.unsqueeze(0).repeat(self.num_layers, 1, 1)
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
            att_tensor = torch.cat(attention_history, dim=-1)
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
