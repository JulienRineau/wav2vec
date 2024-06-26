import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any
import random
from quantization import VectorQuantizer, QuantizerConfig
from loss import Wav2vec2Loss, LossConfig
from wav2vec2 import Wav2Vec2Base, Wav2Vec2Config


class LibriSpeechDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=50000):
        self.num_samples = num_samples
        self.seq_length = seq_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Simulate audio data and its length
        audio = torch.randn(self.seq_length)
        length = random.randint(self.seq_length // 2, self.seq_length)
        return audio, length


class Wav2Vec2PreTraining(pl.LightningModule):
    def __init__(
        self,
        wav2vec_config: Wav2Vec2Config,
        quantizer_config: QuantizerConfig,
        loss_config: LossConfig,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.wav2vec = Wav2Vec2Base(wav2vec_config, quantizer_config)
        self.loss_fn = Wav2vec2Loss(loss_config)

    def training_step(self, batch, batch_idx):
        input_values, lengths = batch
        encoder_out, quantized_features, perplexity, time_mask_indices = self.wav2vec(
            input_values, lengths
        )
        loss = self.loss_fn(
            encoder_out, quantized_features, perplexity, time_mask_indices
        )
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        dataset = LibriSpeechDataset()
        return DataLoader(dataset, batch_size=32, num_workers=4)


if __name__ == "__main__":
    # Configurations
    wav2vec_config = Wav2Vec2Config()
    quantizer_config = QuantizerConfig(
        input_dim=wav2vec_config.n_embd,
        num_groups=2,
        num_choices_per_group=320,
        output_dim=wav2vec_config.n_embd,
        initial_temp=2.0,
    )
    loss_config = LossConfig(
        contrastive_loss_temperature=0.1,
        num_contrastive_loss_negative_samples=100,
        loss_alpha=0.1,
        num_code_vector_groups=2,
        num_code_vectors_per_group=320,
    )

    # Initialize the model
    model = Wav2Vec2PreTraining(wav2vec_config, quantizer_config, loss_config)

    # Initialize a trainer
    trainer = pl.Trainer(
        max_epochs=5,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="value",
    )

    # Train the model
    trainer.fit(model)

    # Test the model
    print("Testing the model...")
    test_input = torch.randn(1, 50000)
    test_length = torch.tensor([45000])

    encoder_out, quantized_features, perplexity, time_mask_indices = model.wav2vec(
        test_input, test_length
    )

    print(f"Encoder output shape: {encoder_out.shape}")
    print(f"Quantized features shape: {quantized_features.shape}")
    print(f"Perplexity shape: {perplexity.shape}")
    print(f"Time mask indices shape: {time_mask_indices.shape}")

    # Compute loss
    loss = model.loss_fn(encoder_out, quantized_features, perplexity, time_mask_indices)
    print(f"Computed loss: {loss.item()}")

    print("All tests passed successfully!")
