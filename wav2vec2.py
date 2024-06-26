import logging
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from quantization import QuantizerConfig, VectorQuantizer


@dataclass
class Wav2Vec2Config:
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 768
    ffn_dim: int = 3072
    max_seq_len: int = 1024
    conv_channels: int = 512
    conv_kernel_sizes: List[int] = (10, 3, 3, 3, 3, 2, 2)
    conv_strides: List[int] = (5, 2, 2, 2, 2, 2, 2)
    dropout: float = 0.1
    pos_conv_kernel: int = 128
    pos_conv_groups: int = 16


class FeatureEncoder(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.config = config

        self.conv_layers = nn.ModuleList()
        in_channels = 1
        for i in range(7):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        config.conv_channels,
                        kernel_size=config.conv_kernel_sizes[i],
                        stride=config.conv_strides[i],
                        padding=(config.conv_kernel_sizes[i] - 1) // 2,
                        bias=False,
                    ),
                    nn.GroupNorm(32, config.conv_channels) if i == 0 else nn.Identity(),
                    nn.GELU(),
                )
            )
            in_channels = config.conv_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        return x.transpose(1, 2)


class FeatureProjection(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_channels)
        self.projection = nn.Linear(config.conv_channels, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.n_embd,
            config.n_embd,
            kernel_size=config.pos_conv_kernel,
            padding="same",
            groups=config.pos_conv_groups,
        )
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        x_conv = self.conv(x_t)
        x_conv = self.gelu(x_conv)
        x = x + x_conv.transpose(1, 2)
        x = self.layer_norm(x)
        return x


class MLP(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.ffn_dim)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.ffn_dim, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        y = self.dropout(y)
        return y


class TransformerBlock(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.attn = SelfAttention(config)
        self.mlp = MLP(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln_1(x)))
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        return x


class Wav2Vec2Base(nn.Module):
    def __init__(self, config: Wav2Vec2Config, quantizer_config: QuantizerConfig):
        super().__init__()
        self.config = config

        self.feature_encoder = FeatureEncoder(config)
        self.feature_projection = FeatureProjection(config)
        self.positional_embedding = PositionalEmbedding(config)
        self.transformer = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        self.quantizer = VectorQuantizer(quantizer_config)
        self.out_linear = nn.Linear(config.n_embd, config.n_embd)

        self.mask_time_prob = 0.065
        self.num_mask_time_steps = 10

    def time_masking(
        self, hidden_states: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        batch_size, num_steps, hidden_size = hidden_states.size()
        time_mask_indices = torch.zeros(
            batch_size,
            num_steps + self.num_mask_time_steps,
            device=hidden_states.device,
            dtype=torch.bool,
        )
        for batch in range(batch_size):
            time_mask_idx_candidates = list(range(int(lengths[batch])))
            k = int(self.mask_time_prob * lengths[batch])
            start_time_idx_array = torch.tensor(
                random.sample(time_mask_idx_candidates, k=k),
                device=hidden_states.device,
            )
            for i in range(self.num_mask_time_steps):
                time_mask_indices[batch, start_time_idx_array + i] = 1
        time_mask_indices = time_mask_indices[:, :num_steps]
        mask_values = torch.zeros_like(hidden_states[time_mask_indices])
        hidden_states[time_mask_indices] = mask_values
        return hidden_states, time_mask_indices

    def forward(
        self, input_values: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor]:
        hidden_states = self.feature_encoder(input_values)
        print(f"Shape after FeatureEncoder: {hidden_states.shape}")

        hidden_states = self.feature_projection(hidden_states)
        print(f"Shape after FeatureProjection: {hidden_states.shape}")

        hidden_states = self.positional_embedding(hidden_states)
        print(f"Shape after PositionalEmbedding: {hidden_states.shape}")

        hidden_states = self.dropout(hidden_states)

        masked_hidden_states, time_mask_indices = self.time_masking(
            hidden_states.clone(), lengths
        )

        quantized_features, perplexity = self.quantizer(hidden_states, lengths)

        for i, block in enumerate(self.transformer):
            masked_hidden_states = block(masked_hidden_states)
            print(f"Shape after TransformerBlock {i}: {masked_hidden_states.shape}")

        encoder_out = self.ln_f(masked_hidden_states)
        encoder_out = self.out_linear(encoder_out)

        return encoder_out, quantized_features, perplexity, time_mask_indices


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    wav2vec_config = Wav2Vec2Config()
    quantizer_config = QuantizerConfig(
        input_dim=wav2vec_config.n_embd,
        num_groups=2,
        num_choices_per_group=320,
        output_dim=wav2vec_config.n_embd,
        initial_temp=2.0,
    )

    model = Wav2Vec2Base(wav2vec_config, quantizer_config).to(device)

    batch_size = 4
    seq_len = 50000
    input_values = torch.randn(batch_size, seq_len).to(device)
    lengths = torch.randint(seq_len // 2, seq_len + 1, (batch_size,)).to(device)

    print(f"Input shape: {input_values.shape}")
    encoder_out, quantized_features, perplexity, time_mask_indices = model(
        input_values, lengths
    )
    print(f"Encoder output shape: {encoder_out.shape}")
    print(f"Quantized features shape: {quantized_features.shape}")
    print(f"Perplexity shape: {perplexity.shape}")
    print(f"Time mask indices shape: {time_mask_indices.shape}")
    print("Custom model test passed successfully!")

    print("\nCustom Model Architecture:")
    print(model)
