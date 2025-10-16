import torch

def random_gain(waveform, min_gain_db=-6, max_gain_db=6):
    gain = torch.rand(1).item() * (max_gain_db - min_gain_db) + min_gain_db
    return waveform * (10 ** (gain / 20))

def add_noise(waveform, noise_level=0.005):
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise
