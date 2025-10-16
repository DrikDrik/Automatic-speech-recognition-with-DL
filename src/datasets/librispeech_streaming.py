import torch
from torch.utils.data import IterableDataset
import torchaudio
import io
import random

from src.transforms.wav_augs import random_gain, add_noise



# THIS DATASET WORKS ONLY WITH STREAMING DATASETS FROM HF
# IT IS VERY MEMORY EFFICIENT AND SUITABLE IN CASE YOU DON'T HAVE TIME TO DOWNLOAD THE WHOLE DATASET LOCALLY

class LibriSpeechTorchDataset(IterableDataset):
    def __init__(self, hf_dataset, tokenizer, length, augment=True):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.augment = augment
        self.length = length
        
    def __len__(self):
        return self.length
     
    def __iter__(self):
        for item in self.dataset:
            audio_info = item["audio"]
            audio_tensor, sr = torchaudio.load(io.BytesIO(audio_info["bytes"])) 
            audio_tensor = audio_tensor.mean(dim=0) 

            if self.augment:
                if random.random() < 0.2:
                    audio_tensor = add_noise(audio_tensor, 0.005)
                if random.random() < 0.3:
                    audio_tensor = random_gain(audio_tensor)


            text_tensor = self.tokenizer(item["text"], return_tensors="pt").input_ids.squeeze(0)
            id = item['id']
            yield {
                "sr": sr,
                "audio": audio_tensor,
                "text": text_tensor,
                "id": id
            }


# THIS ONE BELOW IS DESIGNED TO COMBINE BOTH OTHER AND CLEAN LIBRISPEECH VARIATIONS FOR TRAIN
class CombinedLibriSpeechDataset(IterableDataset):
    def __init__(self, datasets, lengths, tokenizer, shuffle=True):
        self.datasets = datasets
        self.lengths = lengths
        self.tokenizer = tokenizer
        self.shuffle = shuffle
    def __len__(self):
        return sum(self.lengths)
    def __iter__(self):
        iters = [iter(ds) for ds in self.datasets]

        total = sum(self.lengths)
        probs = [l / total for l in self.lengths]

        active = [True] * len(iters)
        while any(active):
            idx = torch.multinomial(torch.tensor(probs), 1).item()

            try:
                yield next(iters[idx])
            except StopIteration:
                active[idx] = False
                probs[idx] = 0
                total = sum(probs)
                if total == 0:
                    break
                probs = [p / total for p in probs]