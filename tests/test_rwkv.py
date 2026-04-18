"""
Test suite for RWKV-7 model implementation.
"""

import torch
import pytest
from src.models.rwkv_model import RWKV7Model, RWKVConfig
from src.models.blocks.rwkv_attention import TimeMixing, ChannelMixing, RWKVLNBlock


class TestRWKVConfig:
    """Test RWKV configuration."""

    def test_default_config(self):
        config = RWKVConfig()
        assert config.name == "RWKV-7"
        assert config.d_model == 512
        assert config.num_layers == 12
        assert config.d_state == 64

    def test_custom_config(self):
        config = RWKVConfig(
            d_model=256,
            num_layers=6,
            vocab_size=32768,
        )
        assert config.d_model == 256
        assert config.num_layers == 6
        assert config.vocab_size == 32768


class TestTimeMixing:
    """Test RWKV Time Mixing layer."""

    def test_time_mixing_forward(self):
        """Test basic forward pass of Time Mixing."""
        batch_size, seq_len, d_model = 2, 16, 64
        d_state = 32

        layer = TimeMixing(d_model, d_state)
        x = torch.randn(batch_size, seq_len, d_model)

        output, last_states = layer(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert last_states.shape == (batch_size, d_state)

    def test_time_mixing_with_init_states(self):
        """Test Time Mixing with provided initial states."""
        batch_size, seq_len, d_model = 2, 8, 64
        d_state = 32

        layer = TimeMixing(d_model, d_state)
        init_states = torch.randn(batch_size, d_state)
        x = torch.randn(batch_size, seq_len, d_model)

        output, last_states = layer(x, init_states)

        assert output.shape == (batch_size, seq_len, d_model)
        assert last_states.shape == (batch_size, d_state)

    def test_time_mixing_single_token(self):
        """Test Time Mixing with single token (no time shift)."""
        batch_size, d_model = 2, 64
        d_state = 32

        layer = TimeMixing(d_model, d_state)
        x = torch.randn(batch_size, 1, d_model)

        output, last_states = layer(x)

        assert output.shape == (batch_size, 1, d_model)
        assert last_states.shape == (batch_size, d_state)


class TestChannelMixing:
    """Test RWKV Channel Mixing layer."""

    def test_channel_mixing_forward(self):
        """Test basic forward pass of Channel Mixing."""
        batch_size, seq_len, d_model = 2, 16, 64

        layer = ChannelMixing(d_model)
        x = torch.randn(batch_size, seq_len, d_model)

        output = layer(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_channel_mixing_with_custom_hidden(self):
        """Test Channel Mixing with custom hidden dimension."""
        batch_size, seq_len, d_model = 2, 16, 64
        d_hidden = 128

        layer = ChannelMixing(d_model, d_hidden)
        x = torch.randn(batch_size, seq_len, d_model)

        output = layer(x)

        assert output.shape == (batch_size, seq_len, d_model)


class TestRWKVLNBlock:
    """Test combined RWKV LN Block."""

    def test_block_forward(self):
        """Test combined block forward pass."""
        batch_size, seq_len, d_model = 2, 16, 64
        d_state = 32

        block = RWKVLNBlock(d_model, d_state)
        x = torch.randn(batch_size, seq_len, d_model)

        output, last_states = block(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert last_states.shape == (batch_size, d_state)

    def test_block_with_hidden_states(self):
        """Test block with provided hidden states."""
        batch_size, seq_len, d_model = 2, 8, 64
        d_state = 32

        block = RWKVLNBlock(d_model, d_state)
        init_states = torch.randn(batch_size, d_state)
        x = torch.randn(batch_size, seq_len, d_model)

        output, last_states = block(x, init_states)

        assert output.shape == (batch_size, seq_len, d_model)


class TestRWKV7Model:
    """Test full RWKV-7 model."""

    def test_model_creation(self):
        """Test model instantiation."""
        config = RWKVConfig(
            vocab_size=1000,
            d_model=64,
            num_layers=4,
            d_state=32,
        )
        model = RWKV7Model(config)

        assert model.config.d_model == 64
        assert model.config.num_layers == 4

    def test_model_forward(self):
        """Test full model forward pass."""
        config = RWKVConfig(
            vocab_size=1000,
            d_model=64,
            num_layers=4,
            d_state=32,
        )
        model = RWKV7Model(config)

        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        logits, final_states = model(input_ids)

        assert logits.shape == (batch_size, seq_len, 1000)
        assert len(final_states) == 4  # num_layers

    def test_model_forward_one_step(self):
        """Test one-step forward for autoregressive inference."""
        config = RWKVConfig(
            vocab_size=1000,
            d_model=64,
            num_layers=4,
            d_state=32,
        )
        model = RWKV7Model(config)

        batch_size = 2
        input_ids = torch.randint(0, 1000, (batch_size, 1))

        logits, new_states = model.forward_one_step(input_ids)

        assert logits.shape == (batch_size, 1, 1000)
        assert len(new_states) == 4

    def test_model_init_hidden_states(self):
        """Test hidden state initialization."""
        config = RWKVConfig(
            vocab_size=1000,
            d_model=64,
            num_layers=4,
            d_state=32,
        )
        model = RWKV7Model(config)

        batch_size = 2
        device = torch.device('cpu')
        states = model.init_hidden_states(batch_size, device)

        assert len(states) == 4
        assert all(s.shape == (batch_size, 32) for s in states)

    def test_model_autoregressive_generation(self):
        """Test simple autoregressive generation."""
        config = RWKVConfig(
            vocab_size=1000,
            d_model=64,
            num_layers=4,
            d_state=32,
        )
        model = RWKV7Model(config)
        model.eval()

        batch_size = 1
        device = next(model.parameters()).device
        states = model.init_hidden_states(batch_size, device)

        # Generate 5 tokens
        generated = []
        current_token = torch.tensor([[1]], device=device)

        with torch.no_grad():
            for _ in range(5):
                logits, states = model.forward_one_step(current_token, states)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs.squeeze(1), 1)
                generated.append(next_token.item())
                current_token = next_token

        assert len(generated) == 5
        assert all(isinstance(t, int) for t in generated)

    def test_model_parameter_count(self):
        """Test parameter counting."""
        config = RWKVConfig(
            vocab_size=1000,
            d_model=64,
            num_layers=4,
            d_state=32,
        )
        model = RWKV7Model(config)

        param_count = model.get_parameter_count()
        assert param_count > 0
        assert isinstance(param_count, int)


class TestRWKVIntegration:
    """Integration tests for RWKV with the transformer-architectures repo."""

    def test_rwkv_import(self):
        """Test that RWKV can be imported from models."""
        from src.models import RWKV7Model, RWKVConfig
        assert RWKV7Model is not None
        assert RWKVConfig is not None

    def test_rwkv_from_config(self):
        """Test creating RWKV model from config."""
        from src.models.rwkv_model import RWKVConfig, RWKV7Model

        config = RWKVConfig(
            d_model=128,
            num_layers=6,
            d_state=64,
        )
        model = RWKV7Model(config)

        assert model.get_parameter_count() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
