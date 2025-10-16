import torch
import matplotlib.pyplot as plt
import numpy as np


def compute_grad_norm(parameters, norm_type=2.0):
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return 0.0
    device_local = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device_local) for p in parameters]), norm_type)
    return total_norm.item()

def plot_mel_to_image(mel_tensor, title=None, n_mels=64):
    if isinstance(mel_tensor, torch.Tensor):
        m = mel_tensor.detach().cpu().numpy()
    else:
        m = np.array(mel_tensor)
    if m.shape[0] < m.shape[1] and m.shape[0] == n_mels:
        pass
    elif m.shape[0] > m.shape[1] and m.shape[1] == n_mels:
        m = m.T
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(m, aspect="auto", origin="lower")
    ax.set_xlabel("time")
    ax.set_ylabel("mel")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return fig

def compute_lr_for_step(global_step, current_epoch, steps_per_epoch, warmup_epochs=1, decay_rate=0.9, start_lr=0.0001, target_lr=0.001):
    warmup_steps = warmup_epochs * steps_per_epoch
    if current_epoch < 25:
        decay_rate = 0.95
    if warmup_steps > 0 and (global_step + 1) <= warmup_steps:
        frac = (global_step + 1) / float(warmup_steps)
        return float(start_lr + frac * (target_lr - start_lr))
    else:
        decay_count = max(0, current_epoch - (warmup_epochs - 1))
        return float(target_lr * (decay_rate ** decay_count))
    
def compute_downsampled_len(length, kernel_size=3, stride=2):
    return ((length - kernel_size) // stride + 1).clamp(min=0)