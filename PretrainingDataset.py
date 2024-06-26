import logging
import os
import random

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset


class StreamingLibriSpeechDataset(IterableDataset):
    def __init__(self, split="train.clean.100", sample_length=250000, debug=False):
        logging.info(f"Initializing StreamingLibriSpeechDataset with split: {split}")
        self.dataset = load_dataset("librispeech_asr", split=split, streaming=True)
        self.sample_length = sample_length
        self.debug = debug

        if self.debug:
            logging.debug("Fetching first item to display keys")
            first_item = next(iter(self.dataset))
            logging.debug("Keys in each datapoint:")
            for key in first_item.keys():
                logging.debug(f"- {key}")

    def __iter__(self):
        logging.info("Starting iteration over the dataset")
        for item in self.dataset:
            audio = torch.tensor(item["audio"]["array"])

            # If audio is shorter than 250k samples, skip it
            if audio.size(0) < self.sample_length:
                if self.debug:
                    logging.debug(f"Skipping short audio: {audio.size(0)} samples")
                continue

            # Randomly crop 250k samples
            if audio.size(0) > self.sample_length:
                start = random.randint(0, audio.size(0) - self.sample_length)
                audio = audio[start : start + self.sample_length]

            if self.debug:
                logging.debug(f"Yielding audio sample with shape: {audio.shape}")

            yield audio, item["audio"]["sampling_rate"]


def collate_fn(batch):
    audio = torch.stack([item[0] for item in batch])
    sampling_rates = [item[1] for item in batch]
    return audio, sampling_rates


def create_dataloader(cuda_available, num_gpus=8, num_workers=4, debug=False):
    logging.info("Creating dataloader")
    dataset = StreamingLibriSpeechDataset(debug=debug)

    if cuda_available:
        # Calculate batch size based on 1.4M samples per GPU
        samples_per_gpu = 1400000
        examples_per_gpu = samples_per_gpu // 250000  # Each example is 250k samples
        total_batch_size = examples_per_gpu * num_gpus
    else:
        # For CPU, use a smaller batch size
        total_batch_size = 4  # Adjust based on your MacBook's capacity

    logging.info(f"Dataloader batch size: {total_batch_size}")
    dataloader = DataLoader(
        dataset,
        batch_size=total_batch_size,
        shuffle=False,
        num_workers=num_workers if cuda_available else 0,
        collate_fn=collate_fn,
        pin_memory=cuda_available,
    )

    return dataloader, total_batch_size


def save_audio_sample(audio, sampling_rate, filename="sample_audio.wav"):
    logging.info(f"Saving audio sample as {filename}")
    audio_np = audio.numpy()
    sf.write(filename, audio_np, sampling_rate)
    logging.info("Audio sample saved successfully")


def main(save_audio=False, debug=False):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.info("Starting main function")
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count() if cuda_available else 0

    logging.info(f"CUDA available: {cuda_available}")
    logging.info(f"Number of GPUs: {num_gpus}")

    dataloader, batch_size_per_gpu = create_dataloader(
        cuda_available, num_gpus, debug=debug
    )

    logging.info(f"Batch size per GPU: {batch_size_per_gpu}")

    logging.info("Starting to iterate over batches")
    for i, (batch, sampling_rates) in enumerate(dataloader):
        logging.info(f"Processing batch {i + 1}")
        if debug:
            logging.debug(f"  Total batch shape: {batch.shape}")
            logging.debug(f"  Batch min: {batch.min()}, max: {batch.max()}")
            logging.debug(f"  Sampling rate: {sampling_rates[0]}")

        if cuda_available:
            gpu_batches = torch.chunk(batch, num_gpus)
            if debug:
                logging.debug(f"  Shape of batch on each GPU: {gpu_batches[0].shape}")

        if i == 0 and save_audio:
            save_audio_sample(batch[0], sampling_rates[0])

        if i == 2:  # Test three batches
            break

    logging.info("Finished processing batches")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("Script started")
    main(save_audio=True, debug=True)
    logging.info("Script completed")
