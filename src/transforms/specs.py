import random
import torch
from torchaudio.transforms import MelSpectrogram


class SpectAugment:
    def __init__(
        self,
        filling_value="mean",
        n_freq_masks=2,
        n_time_masks=2,
        max_freq=3,
        max_time=30,
    ):
        self.filling_value = filling_value
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.max_freq = max_freq
        self.max_time = max_time

    def __call__(self, spectogr, lens):
        spect = spectogr.clone()
        batch_size, n_freq, n_time = spect.shape

        for i in range(batch_size):
            if self.filling_value == "mean":
                v = spect[i].mean()
            elif self.filling_value == "min":
                v = spect[i].min()
            elif self.filling_value == "max":
                v = spect[i].max()
            else:
                v = self.filling_value

            for _ in range(self.n_freq_masks):
                f = random.randint(0, n_freq - 1)
                mask_width = random.randint(1, self.max_freq)
                f_end = min(f + mask_width, n_freq)
                spect[i, f:f_end, :] = v

            for _ in range(self.n_time_masks):
                t = random.randint(0, n_time - 1)
                mask_width = random.randint(1, self.max_time)
                t_end = min(t + mask_width, n_time)
                spect[i, :, t:t_end] = v

        return spect, lens

from torchaudio.transforms import MelSpectrogram
import random

def compute_log_melspectrogram(wav_batch, lens, sr=16000, device="cuda", augment=True):
    wav_batch = wav_batch.to(device)
    featurizer = MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=64,
        center=False,
    ).to(device)
    MS = torch.log(featurizer(wav_batch).clamp(1e-5))
    lens = lens // 256

    if augment and random.random() < 0.5:
        aug = SpectAugment()
        MS, lens = aug(MS, lens)

    return MS, lens