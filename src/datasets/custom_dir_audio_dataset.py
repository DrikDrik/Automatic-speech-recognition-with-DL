from pathlib import Path
import json
import random
import torch
from torch.utils.data import Dataset
import torchaudio
from typing import Dict, Any

from src.transforms.wav_augs import random_gain, add_noise
from src.utils.io_utils import ROOT_PATH


class CustomDirDataset(Dataset):
    def __init__(self, data_dir: str = None, tokenizer=None, augment=False, sr = 16000, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "custom_dir"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.tokenizer = tokenizer
        self.augment = augment
        self._index = self._get_or_load_index()

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self._index[idx]
        audio_path = entry["path"]
        audio_tensor, sr = torchaudio.load(audio_path)
        audio_tensor = audio_tensor.mean(dim=0)

        if self.augment:
            if random.random() < 0.2:
                audio_tensor = add_noise(audio_tensor, 0.005)
            if random.random() < 0.3:
                audio_tensor = random_gain(audio_tensor)

        text = entry.get("text", "")
        text_tensor = self.tokenizer(text, return_tensors="pt").input_ids.squeeze(0)
        item_id = Path(audio_path).stem

        return {
            "sr": sr,
            "audio": audio_tensor,
            "text": text_tensor,
            "id": item_id
        }

    def _get_or_load_index(self):
        index_path = self.data_dir / "index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index()
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self):
        index = []
        audio_dir = self.data_dir / "audio"
        trans_dir = self.data_dir / "transcriptions"

        for path in audio_dir.iterdir():
            if path.suffix.lower() in [".mp3", ".wav", ".flac", ".m4a"]:
                entry = {"path": str(path.absolute().resolve())}
                
                try:
                    t_info = torchaudio.info(entry["path"])
                    entry["audio_len"] = t_info.num_frames / t_info.sample_rate
                except Exception as e:
                    print(f"Warning: Could not get audio info for {path}: {e}")
                    entry["audio_len"] = 0.0
                
                if trans_dir.exists():
                    transc_path = trans_dir / (path.stem + ".txt")
                    if transc_path.exists():
                        with transc_path.open('r', encoding='utf-8') as f:
                            entry["text"] = f.read().strip().lower()
                
                index.append(entry)
        
        return index