import torch
from tqdm.auto import tqdm
import gdown
import os
from src.transforms.specs import compute_log_melspectrogram
from src.utils.utils import compute_downsampled_len
from src.metrics.utils import levenshtein
from src.text_encoder.text import tokens_to_text


class Inferencer:

    def __init__(self, encoder, decoder, config, device, dataloaders, tokenizer, skip_model_load=False, output_dir=None, strategy="beam"):
        assert not skip_model_load or config.inferencer.get("from_pretrained") is not None, \
            "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = config.inferencer
        self.device = device
        self.strategy = strategy
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.tokenizer = tokenizer
        self.dataloaders = dataloaders
        self.beam_size = int(self.cfg_trainer.get("beam_size", 5))
        self.output_dir = output_dir

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        if not skip_model_load:
            encoder_link = "https://drive.google.com/uc?export=download&id=1UpX3_UgrbRTWYunAMHPsR09a1_zHzj7E"
            encoder.load_state_dict(torch.load(gdown.download(encoder_link), map_location='cuda'))
            decoder_link = "https://drive.google.com/uc?export=download&id=1A1Cb1TCn5LWYuIADkOsfvzlBBBi2L2bi"
            decoder.load_state_dict(torch.load(gdown.download(decoder_link), map_location='cuda'))

    def move_batch_to_device(self, batch):
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def transform_batch(self, batch):
        return batch

    def run_inference(self):
        results = {} 
        for part, loader in self.dataloaders.items():
            results[part] = self._inference_part(part, loader)
        return results

    def _inference_part(self, part, dataloader):
        self.encoder.eval()
        self.decoder.eval()

        total_word_edits, total_words = 0, 0
        total_char_edits, total_chars = 0, 0
        sample_table = []
        pad_id_local = 50256

        has_gt = False
        
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader), desc=f"{part}", total=len(dataloader)):
                batch = self.move_batch_to_device(batch)
                batch = self.transform_batch(batch)

                audio = batch["audio"]
                audio_lengths = batch["audio_lengths"]
                sr = batch.get("sr", None)

                B = audio.size(0)
                mels, mel_lens = compute_log_melspectrogram(audio, audio_lengths, sr=sr, device=self.device, augment=False)
                mels = mels.transpose(2, 1)

                padded_len = mels.size(1)
                conv1_padded = compute_downsampled_len(torch.tensor(padded_len, device=self.device))
                max_enc_len = compute_downsampled_len(conv1_padded)
                conv1_lens = compute_downsampled_len(mel_lens)
                enc_lens = compute_downsampled_len(conv1_lens)

                enc_key_padding_mask = torch.ones(B, max_enc_len, device=self.device, dtype=torch.bool)
                for i in range(B):
                    enc_key_padding_mask[i, :enc_lens[i]] = False

                enc_out = self.encoder(mels.unsqueeze(1), key_padding_mask=enc_key_padding_mask)
                if self.strategy == "beam":
                    beam_tensor = self.decoder.beam_search_decode(
                        enc_out, beam_size=self.beam_size, n_best=1, device=self.device, cross_key_padding_mask=None
                    )
                    tokens = beam_tensor[:, 0, :]
                elif self.strategy == "greedy":
                    tokens = self.decoder.autoregressive_decode(
                        enc_out, device=self.device, cross_key_padding_mask=enc_key_padding_mask
                    )

                refs = batch.get("text", batch.get("labels", None))
                has_gt = refs is not None

                for i in range(B):
                    id_ = batch["id"][i] if isinstance(batch["id"], (list, tuple)) else batch["id"][i].item()

                    pred_text = tokens_to_text(tokens[i].cpu(), self.tokenizer, pad_id=pad_id_local).strip()

                    if self.output_dir:
                        part_dir = os.path.join(self.output_dir, part)
                        os.makedirs(part_dir, exist_ok=True)  

                        with open(os.path.join(part_dir, f"pred_ID{id_}.txt"), "w") as f:
                            f.write(pred_text)

                    if has_gt:
                        ref_text = tokens_to_text(refs[i].cpu(), self.tokenizer, pad_id=pad_id_local).strip()

                        total_word_edits += levenshtein(ref_text.split(), pred_text.split())
                        total_words += len(ref_text.split())
                        total_char_edits += levenshtein(list(ref_text), list(pred_text))
                        total_chars += len(ref_text)

                if B > 0 and has_gt:
                    ref_first = tokens_to_text(refs[0].cpu(), self.tokenizer, pad_id=pad_id_local).strip()
                    pred_first = tokens_to_text(tokens[0].cpu(), self.tokenizer, pad_id=pad_id_local).strip()

                    wer_first = levenshtein(ref_first.split(), pred_first.split()) / max(1, len(ref_first.split()))
                    cer_first = levenshtein(list(ref_first), list(pred_first)) / max(1, len(ref_first))
                    sample_table.append({
                        "dataset": part,
                        "batch_idx": batch_idx,
                        "target": ref_first,
                        "predict": pred_first,
                        "wer": wer_first,
                        "cer": cer_first,
                    })

        metrics = {}
        if has_gt:
            wer = total_word_edits / max(1, total_words)
            cer = total_char_edits / max(1, total_chars)
            metrics = {"wer": wer, "cer": cer, "sample_table": sample_table}
            print(f"\n=== {part} FINAL: WER={wer:.4f}, CER={cer:.4f} ===\n")
        return metrics