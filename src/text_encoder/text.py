import torch

def tokens_to_text(token_ids, tokenizer, pad_id=50256):
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    if len(token_ids) == 0:
        return ""
    if pad_id in token_ids:
        token_ids = token_ids[:token_ids.index(pad_id)]
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return text.strip().lower()