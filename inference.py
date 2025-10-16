# inference.py
import warnings
from pathlib import Path
import os

import hydra
import torch
from hydra.utils import instantiate, get_original_cwd

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset, Audio, Features, Value
from transformers import WhisperTokenizer


from src.datasets.librispeech_streaming import LibriSpeechTorchDataset
from src.datasets.custom_dir_audio_dataset import CustomDirDataset

warnings.filterwarnings("ignore", category=UserWarning)


def collate_fn(batch, pad_id=50256):
    audios = [item["audio"] for item in batch]
    audio_lengths = [a.size(0) for a in audios]
    audios_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)  
    text_tensors = []
    any_text = False
    for item in batch:
        t = item.get("text", None)
        if t is None:
            text_tensors.append(None)
        else:
            any_text = True
            if isinstance(t, torch.Tensor):
                text_tensors.append(t.view(-1))
            else:
                text_tensors.append(torch.tensor(t, dtype=torch.long))

    if any_text:
        tensors_for_pad = [t if t is not None else torch.tensor([], dtype=torch.long) for t in text_tensors]
        safe_tensors = []
        zero_mask = []
        for t in tensors_for_pad:
            if t.numel() == 0:
                safe_tensors.append(torch.tensor([pad_id], dtype=torch.long))
                zero_mask.append(True)
            else:
                safe_tensors.append(t)
                zero_mask.append(False)
        texts_padded = pad_sequence(safe_tensors, batch_first=True, padding_value=pad_id)
        for i, zm in enumerate(zero_mask):
            if zm:
                texts_padded[i] = pad_id
    else:
        texts_padded = None

    srs = [item.get("sr", None) for item in batch]
    sr = srs[0] if any(s is not None for s in srs) else None

    ids = [item["id"] for item in batch]

    return {
        "audio": audios_padded,
        "audio_lengths": torch.tensor(audio_lengths, dtype=torch.long),
        "sr": sr,
        "text": texts_padded,
        "id": ids,
    }


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    device = "cuda" if (config.inferencer.device == "auto" and torch.cuda.is_available()) else config.inferencer.device
    text_encoder = instantiate(config.text_encoder)

    tokenizer_name = config.get("tokenizer", {}).get("name", "openai/whisper-small")
    tokenizer = WhisperTokenizer.from_pretrained(tokenizer_name)

    dataloaders = {}

    if config.datasets.get("hf", {}).get("enabled", False):
        features = Features(
            {
                "file": Value("string"),
                "audio": Audio(decode=False),
                "text": Value("string"),
                "speaker_id": Value("int64"),
                "chapter_id": Value("int64"),
                "id": Value("string"),
            }
        )
        hf = config.datasets.hf

        clean_ds_stream = load_dataset("librispeech_asr", "clean", split="test", features=features, streaming=True)
        other_ds_stream = load_dataset("librispeech_asr", "other", split="test", features=features, streaming=True)

        dataset_clean_test = LibriSpeechTorchDataset(clean_ds_stream, tokenizer, length=2620, augment=False)
        dataset_other_test = LibriSpeechTorchDataset(other_ds_stream, tokenizer, length=2939, augment=False)

        bs = int(config.inferencer.get("batch_size", 16))
        num_workers = int(config.dataloader.get("num_workers", 4))
        pad_id = int(config.inferencer.get("pad_id", 50256))

        dataloaders["test_clean"] = DataLoader(
            dataset_clean_test,
            batch_size=bs,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=(device == "cuda"),
        )

        dataloaders["test_other"] = DataLoader(
            dataset_other_test,
            batch_size=bs,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=(device == "cuda"),
        )

    # BELOW FOR CUSTOM
    if config.datasets.get("custom_dir", {}).get("enabled", False):
        cd_cfg = config.datasets.custom_dir
        custom_path = cd_cfg.get("path", None)
        cd_dataset = CustomDirDataset(data_dir=custom_path, tokenizer=tokenizer, augment=False)
        bs = int(config.inferencer.get("batch_size", 8))
        num_workers = int(config.dataloader.get("num_workers", 4))
        pad_id = int(config.inferencer.get("pad_id", 50256))

        dataloaders["custom"] = DataLoader(
            cd_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
            pin_memory=(device == "cuda"),
        )

    if len(dataloaders) == 0:
        raise RuntimeError("No dataloaders were created. Enable at least one dataset in src/configs/inference.yaml under datasets.hf or datasets.custom_dir")

    encoder = instantiate(config.model.encoder).to(device)
    decoder = instantiate(config.model.decoder).to(device)

    output_dir = config.inferencer.get("output_dir", "predictions_infer")
    base_out = Path(get_original_cwd()) / output_dir
    base_out.mkdir(parents=True, exist_ok=True)

    from src.trainer import Inferencer

    inferencer = Inferencer(
        encoder=encoder,
        decoder=decoder,
        config=config,
        device=device,
        dataloaders=dataloaders,
        text_encoder=text_encoder,
        skip_model_load=bool(config.inferencer.get("skip_model_load", False)),
        output_dir=str(base_out),
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:25s}: {value}")

    print(f"Predictions saved to: {base_out}")


if __name__ == "__main__":
    main()
