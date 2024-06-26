import logging
import random

import soundfile as sf
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FineTuningLibriSpeechDataset(IterableDataset):
    def __init__(self, split="train.clean.100", sample_length=250000, debug=False):
        logging.info(f"Initializing FineTuningLibriSpeechDataset with split: {split}")
        self.dataset = load_dataset("librispeech_asr", split=split, streaming=True)
        self.sample_length = sample_length
        self.debug = debug

        if self.debug:
            first_item = next(iter(self.dataset))
            logging.info(f"Dataset keys: {', '.join(first_item.keys())}")

    def __iter__(self):
        for item in self.dataset:
            audio = torch.tensor(item["audio"]["array"])
            transcription = item["text"]

            if audio.size(0) < self.sample_length:
                audio = torch.nn.functional.pad(
                    audio, (0, self.sample_length - audio.size(0))
                )
            elif audio.size(0) > self.sample_length:
                start = random.randint(0, audio.size(0) - self.sample_length)
                audio = audio[start : start + self.sample_length]

            yield audio, item["audio"]["sampling_rate"], transcription


def collate_fn(batch):
    audio = torch.stack([item[0] for item in batch])
    sampling_rates = [item[1] for item in batch]
    transcriptions = [item[2] for item in batch]
    return audio, sampling_rates, transcriptions


def create_fine_tuning_dataloader(batch_size=16, debug=False):
    dataset = FineTuningLibriSpeechDataset(debug=debug)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return dataloader


def save_audio_sample(
    audio, sampling_rate, transcription, filename="fine_tuning_sample.wav"
):
    audio_np = audio.numpy()
    sf.write(filename, audio_np, sampling_rate)

    with open(f"{filename}.txt", "w") as f:
        f.write(transcription)

    logging.info(f"Saved audio sample as {filename} with transcription")


def main(debug=False):
    logging.info("Starting fine-tuning dataset preparation")

    dataloader = create_fine_tuning_dataloader(debug=debug)

    for i, (batch, sampling_rates, transcriptions) in enumerate(dataloader):
        logging.info(f"Batch {i + 1}:")
        logging.info(f"  Audio shape: {batch.shape}")
        logging.info(f"  Sampling rate: {sampling_rates[0]}")
        logging.info(f"  Transcription sample: {transcriptions[0][:50]}...")

        if i == 0:
            save_audio_sample(
                batch[0],
                sampling_rates[0],
                transcriptions[0],
                f"fine_tuning_sample_{i}.wav",
            )

        if i == 2:  # Process three batches
            break

    logging.info("Finished processing fine-tuning dataset samples")


if __name__ == "__main__":
    main(debug=True)
