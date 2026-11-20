"""
Unit tests for Proposed KAN-Enhanced Seq2Seq LSTM Autoencoder.
Verifies KANLinear layer, B-spline evaluation, full sequence model forward pass,
reconstruction shapes, attention extraction, loss gradient flow across KAN parameters,
optimizer weight updates, checkpoint persistence, variable D/T dimensions, and factory integration.
"""

from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn

from src.utils.config_loader import load_research_config
from src.models.kan_layer import KANLinear
from src.models.kan_seq2seq import KANSeq2SeqAutoencoder
from src.models import build_model
from src.models.trainer import train_autoencoder, load_checkpoint


class TestKANModel(unittest.TestCase):
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

    def test_kan_layer_initialization(self):
        layer = KANLinear(in_features=20, out_features=10, grid_size=5, spline_order=3)
        self.assertEqual(layer.in_features, 20)
        self.assertEqual(layer.out_features, 10)
        self.assertEqual(layer.grid_size, 5)
        self.assertEqual(layer.spline_order, 3)
        self.assertIsNotNone(layer.base_weight)
        self.assertIsNotNone(layer.spline_weight)
        self.assertIsNotNone(layer.spline_scaler)

    def test_kan_layer_forward_and_output_dimensions(self):
        layer = KANLinear(in_features=16, out_features=8, grid_size=5, spline_order=3)
        x = torch.randn(4, 16)
        out = layer(x)
        self.assertEqual(out.shape, (4, 8))

    def test_kan_seq2seq_model_initialization(self):
        model = KANSeq2SeqAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=2,
            dropout=0.2,
            kan_grid_size=5,
            kan_spline_order=3,
        )
        self.assertEqual(model.input_dim, self.input_dim)
        self.assertEqual(model.hidden_dim, self.hidden_dim)
        self.assertEqual(model.latent_dim, self.latent_dim)

    def test_kan_sequence_model_forward_and_reconstruction_shape(self):
        model = KANSeq2SeqAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        )
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        recon = model(x, teacher_forcing_ratio=0.0)
        self.assertEqual(recon.shape, (self.batch_size, self.seq_len, self.input_dim))

    def test_kan_anomaly_score_shape(self):
        model = KANSeq2SeqAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        )
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        scores = model.compute_anomaly_score(x, aggregation="mean")
        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(scores.shape, (self.batch_size,))
        self.assertTrue(np.all(scores >= 0))

    def test_kan_attention_output(self):
        model = KANSeq2SeqAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        )
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        recon, att = model(x, return_attention=True)
        self.assertEqual(recon.shape, (self.batch_size, self.seq_len, self.input_dim))
        self.assertEqual(att.shape, (self.batch_size, self.seq_len, self.seq_len))
        sums = torch.sum(att, dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_kan_backward_pass_and_gradients(self):
        model = KANSeq2SeqAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        optimizer.zero_grad()
        recon = model(x, teacher_forcing_ratio=0.5)
        loss = criterion(recon, x)
        loss.backward()

        # Explicitly verify KAN parameters receive non-zero gradients
        self.assertIsNotNone(model.kan_encoder_bottleneck.base_weight.grad)
        self.assertIsNotNone(model.kan_encoder_bottleneck.spline_weight.grad)
        self.assertIsNotNone(model.kan_encoder_bottleneck.spline_scaler.grad)
        self.assertIsNotNone(model.kan_decoder_bottleneck.base_weight.grad)
        self.assertIsNotNone(model.kan_decoder_bottleneck.spline_weight.grad)

        self.assertGreater(model.kan_encoder_bottleneck.spline_weight.grad.norm().item(), 0.0)
        self.assertGreater(model.kan_decoder_bottleneck.spline_weight.grad.norm().item(), 0.0)

    def test_kan_optimizer_update(self):
        model = KANSeq2SeqAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        orig_spline_weights = model.kan_encoder_bottleneck.spline_weight.clone()
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        
        optimizer.zero_grad()
        recon = model(x)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()

        # Weights must be updated
        self.assertFalse(torch.allclose(orig_spline_weights, model.kan_encoder_bottleneck.spline_weight))

    def test_kan_checkpoint_save_and_load(self):
        model = KANSeq2SeqAutoencoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim)
        ckpt_dir = Path(self.test_dir) / "checkpoints/kan_seq2seq"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file = ckpt_dir / "test_kan_model.pt"

        torch.save({
            "epoch": 4,
            "model_state_dict": model.state_dict(),
            "val_loss": 0.0456,
            "input_dim": self.input_dim,
        }, ckpt_file)

        fresh_model = KANSeq2SeqAutoencoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim)
        fresh_model, loaded_ckpt = load_checkpoint(str(ckpt_file), fresh_model)
        self.assertEqual(loaded_ckpt["epoch"], 4)
        self.assertAlmostEqual(loaded_ckpt["val_loss"], 0.0456)

    def test_kan_different_feature_dimensions(self):
        for d in [5, 11, 48, 80]:
            m = KANSeq2SeqAutoencoder(input_dim=d, hidden_dim=32, latent_dim=16)
            inp = torch.randn(2, 30, d)
            out = m(inp)
            self.assertEqual(out.shape, (2, 30, d))

    def test_kan_different_sequence_lengths(self):
        for seq_l in [10, 30, 50]:
            m = KANSeq2SeqAutoencoder(input_dim=self.input_dim, hidden_dim=32, latent_dim=16)
            inp = torch.randn(2, seq_l, self.input_dim)
            out = m(inp)
            self.assertEqual(out.shape, (2, seq_l, self.input_dim))

    def test_kan_cpu_execution(self):
        model = KANSeq2SeqAutoencoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim).to("cpu")
        x = torch.randn(2, 15, self.input_dim).to("cpu")
        out = model(x)
        self.assertEqual(out.device.type, "cpu")

    def test_kan_factory_compatibility(self):
        model = build_model("kan_seq2seq", input_dim=self.input_dim, config=self.config)
        self.assertIsInstance(model, KANSeq2SeqAutoencoder)

    def test_kan_no_label_dependency_in_forward(self):
        model = KANSeq2SeqAutoencoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim)
        x = torch.randn(4, 25, self.input_dim)
        out = model(x)
        self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
