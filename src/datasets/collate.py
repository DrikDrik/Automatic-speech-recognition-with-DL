from torch.nn.utils.rnn import pad_sequence
import torch

def collate_fn(batch):
    pad_id = 50256
    audios = [item["audio"] for item in batch]
    audio_lengths = [a.size(0) for a in audios]
    audios_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)
    texts = [item["text"] for item in batch]
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=pad_id)
    srs = [item["sr"] for item in batch]
    
    return {
        "audio": audios_padded,
        "text": texts_padded,
        'sr': srs[0],
        'audio_lengths': torch.tensor(audio_lengths),
        'len': audios_padded.shape[-1]
    }
