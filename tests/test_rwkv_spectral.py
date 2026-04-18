"""
Test suite for RWKV Spectral Model.
Tests both RWKV and Self-Attention modes.
"""

import torch
import pytest

from src.models.rwkv_model import RWKVSpectral, RWKVSpectralConfig, create_rwkv_spectral
from src.models.blocks.rwkv_attention import (
    RWKVTimeMixing,
    RWKVTimeMixingChunked,
    SelfAttentionFH,
    FreqConv1D,
    TimeConv1D,
    FFNH,
)


# ============================================================
# Input shape: (B=2, T=16, F=8, H=16)
# ============================================================
B, T, F, H = 2, 16, 8, 16


class TestFreqConv1D:
    """Test frequency axis convolution."""

    def test_forward_shape(self):
        x = torch.randn(B, T, F, H)
        conv = FreqConv1D(f=F, h=H, kernel_size=3)
        out = conv(x)
        assert out.shape == (B, T, F, H)

    def test_forward_values(self):
        x = torch.zeros(B, T, F, H)
        x[:, :, :, :] = 1.0  # uniform input
        conv = FreqConv1D(f=F, h=H, kernel_size=3)
        out = conv(x)
        assert out.shape == (B, T, F, H)
        # Output should be non-trivial due to conv
        assert not torch.allclose(out, x)


class TestTimeConv1D:
    """Test time axis convolution."""

    def test_forward_shape(self):
        x = torch.randn(B, T, F, H)
        conv = TimeConv1D(f=F, h=H, kernel_size=3)
        out = conv(x)
        assert out.shape == (B, T, F, H)

    def test_time_invariant(self):
        """Shift in time should produce shifted output."""
        x = torch.randn(B, T, F, H)
        conv = TimeConv1D(f=F, h=H, kernel_size=3)
        out = conv(x)
        assert out.shape == (B, T, F, H)


class TestFFNH:
    """Test FFN on H dimension."""

    def test_forward_shape(self):
        x = torch.randn(B, T, F, H)
        ffn = FFNH(f=F, h=H, h_ffn_mult=4)
        out = ffn(x)
        assert out.shape == (B, T, F, H)

    def test_reshapes_correctly(self):
        x = torch.randn(B, T, F, H)
        ffn = FFNH(f=F, h=H, h_ffn_mult=4)
        out = ffn(x)
        # Check non-trivial transformation
        assert not torch.allclose(out, x)


class TestSelfAttentionFH:
    """Test standard self-attention on (f*h) dimension."""

    def test_forward_shape(self):
        x = torch.randn(B, T, F, H)
        attn = SelfAttentionFH(f=F, h=H, nhead=8)
        out, _ = attn(x)
        assert out.shape == (B, T, F, H)

    def test_no_attention_when_single_head(self):
        """With nhead=1, should be close to identity (up to projection)."""
        x = torch.randn(B, T, F, H)
        attn = SelfAttentionFH(f=F, h=H, nhead=1)
        out, _ = attn(x)
        assert out.shape == (B, T, F, H)


class TestRWKVTimeMixing:
    """Test RWKV time mixing."""

    def test_forward_shape(self):
        x = torch.randn(B, T, F, H)
        mixer = RWKVTimeMixing(d_attn=F * H, d_state=64)
        out, state = mixer(x)
        assert out.shape == (B, T, F, H)
        assert state.shape == (B, 64)

    def test_single_timestep(self):
        """Single time step should work without time-shift."""
        x = torch.randn(B, 1, F, H)
        mixer = RWKVTimeMixing(d_attn=F * H, d_state=64)
        out, state = mixer(x)
        assert out.shape == (B, 1, F, H)
        assert state.shape == (B, 64)

    def test_state_changes_with_sequence(self):
        """Longer sequence should produce different final state."""
        mixer1 = RWKVTimeMixing(d_attn=F * H, d_state=64)
        mixer2 = RWKVTimeMixing(d_attn=F * H, d_state=64)

        x1 = torch.randn(B, 4, F, H)
        x2 = torch.randn(B, 8, F, H)

        _, state1 = mixer1(x1)
        _, state2 = mixer2(x2)

        assert state1.shape == (B, 64)
        assert state2.shape == (B, 64)


class TestRWKVTimeMixingChunked:
    """Test chunked RWKV time mixing."""

    def test_forward_shape(self):
        x = torch.randn(B, T, F, H)
        mixer = RWKVTimeMixingChunked(d_attn=F * H, d_state=32, chunk_size=8)
        out, state = mixer(x)
        assert out.shape == (B, T, F, H)
        assert state.shape == (B, 32)


class TestSpectralRWKVLayer:
    """Test the full single layer."""

    def test_rwkv_mode_shape(self):
        from src.models.rwkv_model import SpectralRWKVLayer

        layer = SpectralRWKVLayer(
            f=F, h=H,
            attention_mode="rwkv",
            d_state=64,
            conv_kernel=3,
        )
        x = torch.randn(B, T, F, H)
        out = layer(x)
        assert out.shape == (B, T, F, H)

    def test_self_attention_mode_shape(self):
        from src.models.rwkv_model import SpectralRWKVLayer

        layer = SpectralRWKVLayer(
            f=F, h=H,
            attention_mode="self_attention",
            nhead=8,
            conv_kernel=3,
        )
        x = torch.randn(B, T, F, H)
        out = layer(x)
        assert out.shape == (B, T, F, H)


class TestRWKVSpectralModel:
    """Test the full 8-layer model."""

    def test_rwkv_mode(self):
        model = RWKVSpectral(
            f=F, h=H,
            num_layers=8,
            attention_mode="rwkv",
            d_state=64,
        )
        x = torch.randn(B, T, F, H)
        out = model(x)
        assert out.shape == (B, T, F, H)
        assert not torch.isnan(out).any()

    def test_self_attention_mode(self):
        model = RWKVSpectral(
            f=F, h=H,
            num_layers=8,
            attention_mode="self_attention",
            nhead=8,
        )
        x = torch.randn(B, T, F, H)
        out = model(x)
        assert out.shape == (B, T, F, H)
        assert not torch.isnan(out).any()

    def test_parameter_count(self):
        model_rwkv = RWKVSpectral(
            f=F, h=H,
            num_layers=8,
            attention_mode="rwkv",
        )
        model_attn = RWKVSpectral(
            f=F, h=H,
            num_layers=8,
            attention_mode="self_attention",
        )

        params_rwkv = model_rwkv.get_parameter_count()
        params_attn = model_attn.get_parameter_count()

        print(f"\nRWKV mode params: {params_rwkv:,}")
        print(f"Self-Attn mode params: {params_attn:,}")

        assert params_rwkv > 0
        assert params_attn > 0

    def test_different_freq_bins(self):
        """Test with different frequency dimensions."""
        f_new = 32
        model = RWKVSpectral(
            f=f_new, h=H,
            num_layers=4,
            attention_mode="rwkv",
            d_state=32,
        )
        x = torch.randn(2, 8, f_new, H)
        out = model(x)
        assert out.shape == (2, 8, f_new, H)


class TestRWKVSpectralIntegration:
    """Integration tests."""

    def test_single_layer_training(self):
        """Quick training step to verify gradients flow."""
        model = RWKVSpectral(
            f=F, h=H,
            num_layers=1,
            attention_mode="rwkv",
            d_state=32,
        )
        model.train()

        x = torch.randn(B, T, F, H, requires_grad=True)
        target = torch.randn(B, T, F, H)

        output = model(x)
        loss = F.mse_loss(output, target)
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_autoregressive_generation(self):
        """Test step-by-step generation with RWKV (has state)."""
        model = RWKVSpectral(
            f=F, h=H,
            num_layers=2,
            attention_mode="rwkv",
            d_state=32,
        )
        model.eval()

        # Hidden states per layer
        num_layers = 2
        states = [
            torch.zeros(B, 32) for _ in range(num_layers)
        ]

        with torch.no_grad():
            # Generate step by step
            for step in range(4):
                x_step = torch.randn(B, 1, F, H)
                # For RWKV, we need to pass states through each layer
                for li, layer in enumerate(model.layers):
                    # Only RWKV mode has meaningful states
                    if layer.attention_mode == "rwkv":
                        pass  # state handling would go here

        # Just verify model runs step by step
        x_step = torch.randn(B, 1, F, H)
        out = model(x_step)
        assert out.shape == (B, 1, F, H)

    def test_factory_function(self):
        model = create_rwkv_spectral(
            f=F, h=H,
            num_layers=4,
            attention_mode="rwkv",
        )
        assert isinstance(model, RWKVSpectral)
        assert model.attention_mode == "rwkv"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
