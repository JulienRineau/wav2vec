import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple


@dataclass
class QuantizerConfig:
    input_dim: int
    num_groups: int
    num_choices_per_group: int
    output_dim: int
    initial_temp: float


class GumbelVectorQuantizer(nn.Module):
    def __init__(self, config: QuantizerConfig):
        super().__init__()
        self.num_groups = config.num_groups
        self.num_choices = config.num_choices_per_group
        self.output_dim = config.output_dim

        self.projector = nn.Linear(config.input_dim, self.num_groups * self.num_choices)

        self.codebook = nn.Parameter(
            torch.FloatTensor(
                self.num_groups, self.num_choices, config.output_dim // self.num_groups
            ).uniform_(-1 / self.num_choices, 1 / self.num_choices)
        )

        self.temperature = nn.Parameter(torch.tensor(config.initial_temp))

    @staticmethod
    def compute_usage(
        probabilities: torch.Tensor, valid_lengths: torch.Tensor
    ) -> torch.Tensor:
        mask = (
            torch.arange(probabilities.size(1), device=probabilities.device)[None, :]
            < valid_lengths[:, None]
        )
        masked_sum = torch.sum(
            probabilities * mask.unsqueeze(-1).unsqueeze(-1), dim=(0, 1)
        )
        normalizer = torch.sum(mask.float())
        usage = masked_sum / normalizer
        return usage

    def forward(
        self, inputs: torch.Tensor, valid_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length, _ = inputs.shape

        # Project inputs
        logits = self.projector(inputs).view(
            batch_size, seq_length, self.num_groups, self.num_choices
        )

        # Apply Gumbel softmax
        choice_probs = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)

        # Compute usage statistics
        soft_probs = F.softmax(logits, dim=-1)
        usage = self.compute_usage(soft_probs, valid_lengths)

        # Quantize
        quantized = torch.einsum("bsgc,gcd->bsgd", choice_probs, self.codebook)
        quantized = quantized.reshape(batch_size, seq_length, self.output_dim)

        return quantized, usage


if __name__ == "__main__":
    torch.manual_seed(42)

    config = QuantizerConfig(
        input_dim=256,
        num_groups=2,
        num_choices_per_group=320,
        output_dim=128,
        initial_temp=2.0,
    )

    quantizer = GumbelVectorQuantizer(config)

    batch_size = 64
    seq_length = 100
    inputs = torch.randn(batch_size, seq_length, config.input_dim)
    valid_lengths = torch.randint(1, seq_length + 1, (batch_size,))

    quantized, usage = quantizer(inputs, valid_lengths)

    print(f"Input shape: {inputs.shape}")
    print(f"Quantized output shape: {quantized.shape}")
    print(f"Usage statistics shape: {usage.shape}")

    assert quantized.shape == (batch_size, seq_length, config.output_dim)
    assert usage.shape == (config.num_groups, config.num_choices_per_group)

    print("All tests passed successfully!")
