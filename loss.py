import random
from dataclasses import dataclass

import torch
from torch import nn
from torch.autograd import Variable


@dataclass(frozen=True)
class LossConfig:
    contrastive_loss_temperature: float
    num_contrastive_loss_negative_samples: int
    loss_alpha: float
    num_code_vector_groups: int
    num_code_vectors_per_group: int


class Wav2vec2Loss(nn.Module):
    def __init__(self, config: LossConfig):
        super().__init__()
        self.k = config.contrastive_loss_temperature
        self.K = config.num_contrastive_loss_negative_samples
        self.a = config.loss_alpha
        self.G = config.num_code_vector_groups
        self.V = config.num_code_vectors_per_group
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, encoder_out, quantized_features, perplexity, time_mask_indices):
        target_encoder_out = encoder_out[time_mask_indices]
        labels = quantized_features[time_mask_indices]

        # Make negative samples
        negative_samples = self.negative_sampler(labels)
        negative_samples = torch.cat([labels.unsqueeze(1), negative_samples], dim=1)

        contrastive_loss = self.contrastive_loss(
            target_encoder_out, labels, negative_samples
        )
        diversity_loss = self.diversity_loss(perplexity)

        loss = contrastive_loss + self.a * diversity_loss
        return loss

    def contrastive_loss(
        self,
        targets: torch.Tensor,
        labels: torch.Tensor,
        negative_samples: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            targets (torch.Tensor): with shape `(N, D)`
            labels (torch.Tensor): with shape `(N, D)`
            negative_samples (torch.Tensor): with shape `(N, K, D)`

        Returns:
            torch.Tensor with shape `(1)`
        """
        pos_sim = self.cos(targets, labels)
        neg_sim = self.cos(targets.unsqueeze(1), negative_samples)
        pos_sim_exp = torch.exp(pos_sim / self.k)
        neg_sim_exp = torch.exp(neg_sim / self.k).sum(dim=1)
        return -torch.log(pos_sim_exp / neg_sim_exp).mean()

    def diversity_loss(self, perplexity):
        """
        Compute the diversity loss based on the entropy of perplexity values.
        Args:
            perplexity (torch.Tensor): with shape `(G, V)`, where `G` is the number of groups,
                                    and `V` is the number of vectors (entries) per group.
        Returns:
            torch.Tensor: Scalar representing the average entropy across all groups and entries.
        """
        # Add a small constant for numerical stability in log computation
        log_perplexity = torch.log(perplexity + 1e-9)
        entropy = -(
            perplexity * log_perplexity
        ).sum()  # Sum entropy over all groups and entries

        # Average entropy over the number of groups and entries
        average_entropy = entropy / (self.G * self.V)

        return average_entropy

    def negative_sampler(self, labels: torch.Tensor):
        """
        Args:
            labels (torch.Tensor): with shape `(N, D)`

        Returns:
            torch.Tensor with shape `(N, K, D)'

        """
        batch_size, _ = labels.shape
        negative_samples = []
        for b in range(batch_size):
            negatives = torch.cat([labels[:b], labels[b + 1 :]])
            indices = torch.randperm(negatives.size(0))[: self.K]
            negative_samples.append(negatives[indices])
        return torch.stack(negative_samples, dim=0)


if __name__ == "__main__":

    class TestLossConfig:
        contrastive_loss_temperature = 0.1
        num_contrastive_loss_negative_samples = 10
        loss_alpha = 0.1
        num_code_vector_groups = 4
        num_code_vectors_per_group = 10

    config = TestLossConfig()
    loss_module = Wav2vec2Loss(config)

    batch_size = 5
    feature_dim = 128
    num_time_mask_indices = 3

    encoder_out = Variable(torch.rand(batch_size, feature_dim), requires_grad=True)
    quantized_features = Variable(
        torch.rand(batch_size, feature_dim), requires_grad=False
    )
    perplexity = Variable(
        torch.rand(config.num_code_vector_groups, config.num_code_vectors_per_group),
        requires_grad=False,
    )
    time_mask_indices = Variable(
        torch.randint(0, batch_size, (num_time_mask_indices,)), requires_grad=False
    )

    loss = loss_module(encoder_out, quantized_features, perplexity, time_mask_indices)
    print("Computed Loss:", loss.item())

    assert loss.requires_grad, "Loss tensor should require gradients"
