"""
Unit tests for Baseline Seq2Seq LSTM Autoencoder with Bahdanau Attention.
Verifies initialization, forward pass, reconstruction shapes, attention extraction,
loss calculation, backward pass, checkpoint save/load, variable D/T shapes, and label independence.
"""

from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn

from src.utils.config_loader import load_research_config
from src.models.baseline_seq2seq import BaselineSeq2SeqAutoencoder, BahdanauAttention
from src.models import build_model
from src.models.summary import summarize_model
from src.models.trainer import train_autoencoder, load_checkpoint


class TestBaselineModel(unittest.TestCase):
    def setUp(self):
        self.config = load_research_config("configs/research.yaml")
        self.test_dir = tempfile.mkdtemp()
        self.batch_size = 8
        self.seq_len = 30
        self.input_dim = 16
        self.hidden_dim = 32
        self.latent_dim = 16

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_model_initialization(self):
        model = BaselineSeq2SeqAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=2,
            dropout=0.2
        )
        self.assertEqual(model.input_dim, self.input_dim)
        self.assertEqual(model.hidden_dim, self.hidden_dim)
        self.assertEqual(model.latent_dim, self.latent_dim)

    def test_forward_pass_and_output_shapes(self):
        model = BaselineSeq2SeqAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        )
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        recon = model(x, teacher_forcing_ratio=0.0)
        self.assertEqual(recon.shape, (self.batch_size, self.seq_len, self.input_dim))

    def test_reconstruction_shape(self):
        model = build_model("baseline_seq2seq", input_dim=self.input_dim, config=self.config)
        x = torch.randn(4, 20, self.input_dim)
        recon = model(x)
        self.assertEqual(recon.shape, x.shape)

    def test_anomaly_score_shape(self):
        model = BaselineSeq2SeqAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        )
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        scores = model.compute_anomaly_score(x, aggregation="mean")
        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(scores.shape, (self.batch_size,))
        self.assertTrue(np.all(scores >= 0))

    def test_attention_output_and_influence(self):
        model = BaselineSeq2SeqAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        )
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        recon, att_weights = model(x, return_attention=True)
        self.assertEqual(recon.shape, (self.batch_size, self.seq_len, self.input_dim))
        self.assertEqual(att_weights.shape, (self.batch_size, self.seq_len, self.seq_len))
        # Attention weights should sum to 1 across key sequence
        sums = torch.sum(att_weights, dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_loss_calculation_and_backward_pass(self):
        model = BaselineSeq2SeqAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        x = torch.randn(self.batch_size, self.seq_len, self.input_dim, requires_grad=False)
        optimizer.zero_grad()
        recon = model(x, teacher_forcing_ratio=0.5)
        loss = criterion(recon, x)
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        # Verify gradients exist on encoder, attention, bottleneck, and decoder
        self.assertIsNotNone(model.encoder_lstm.weight_ih_l0.grad)
        self.assertIsNotNone(model.attention.W_query.weight.grad)
        self.assertIsNotNone(model.encoder_bottleneck.weight.grad)
        self.assertIsNotNone(model.decoder_lstm.weight_ih_l0.grad)
        self.assertIsNotNone(model.reconstruction_head.weight.grad)

    def test_checkpoint_save_and_load(self):
        model = BaselineSeq2SeqAutoencoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim)
        ckpt_dir = Path(self.test_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file = ckpt_dir / "test_model.pt"

        # Save checkpoint
        torch.save({
            "epoch": 5,
            "model_state_dict": model.state_dict(),
            "val_loss": 0.1234,
            "input_dim": self.input_dim,
        }, ckpt_file)

        # Reload checkpoint into fresh model
        fresh_model = BaselineSeq2SeqAutoencoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim)
        fresh_model, loaded_ckpt = load_checkpoint(str(ckpt_file), fresh_model)
        self.assertEqual(loaded_ckpt["epoch"], 5)
        self.assertAlmostEqual(loaded_ckpt["val_loss"], 0.1234)

    def test_different_feature_counts(self):
        for d in [5, 11, 48, 80]:
            m = BaselineSeq2SeqAutoencoder(input_dim=d, hidden_dim=32, latent_dim=16)
            inp = torch.randn(2, 30, d)
            out = m(inp)
            self.assertEqual(out.shape, (2, 30, d))

    def test_different_sequence_lengths(self):
        for seq_l in [10, 30, 50]:
            m = BaselineSeq2SeqAutoencoder(input_dim=self.input_dim, hidden_dim=32, latent_dim=16)
            inp = torch.randn(2, seq_l, self.input_dim)
            out = m(inp)
            self.assertEqual(out.shape, (2, seq_l, self.input_dim))

    def test_cpu_execution(self):
        model = BaselineSeq2SeqAutoencoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim).to("cpu")
        x = torch.randn(2, 15, self.input_dim).to("cpu")
        out = model(x)
        self.assertEqual(out.device.type, "cpu")

    def test_no_label_dependency_in_forward(self):
        # Forward pass signature must accept only input tensors and structural hyperparameters
        model = BaselineSeq2SeqAutoencoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim)
        x = torch.randn(4, 25, self.input_dim)
        # Should execute purely on x without requiring any labels
        out = model(x)
        self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
