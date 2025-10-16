import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import random
from tqdm import tqdm
import gc
import wandb
import matplotlib.pyplot as plt
from src.transforms.specs import compute_log_melspectrogram
from src.utils.utils import compute_lr_for_step
from src.utils.utils import compute_downsampled_len
from src.metrics.utils import levenshtein
from src.text_encoder.text import tokens_to_text
from src.utils.utils import compute_grad_norm, plot_mel_to_image


class Trainer:
    def __init__(
        self,
        encoder,
        decoder,
        tokenizer,
        train_loader,
        loader_val_clean,
        loader_val_other,
        device = "cuda",
        pad_id = 50256,
        eos_id = 50257,
        bos_id = 50258,
        notimestamps_id = 50363,
        vocab_size = 51865,
        start_lr = 1e-4,
        num_epochs = 100,
        log_interval = 100,
        examples_per_log = 3,
        eval_batches = 10,
        run_name = "train",
        encoder_save_path='/content/encoder.pth',
        decoder_save_path='/content/decoder.pth'
    ):
        wandb.init(project="asr-training", name=run_name) 
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

        self.encoder_save_path = encoder_save_path  
        self.decoder_save_path = decoder_save_path

        self.tokenizer = tokenizer

        self.train_loader = train_loader
        self.loader_val_clean = loader_val_clean
        self.loader_val_other = loader_val_other

        self.device = device
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.notimestamps_id = notimestamps_id
        self.vocab_size = vocab_size

        
        weights = torch.ones(vocab_size, device=device)
        weights[eos_id] = 2.0
        self.loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=pad_id)
        self.optimizer = Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=start_lr)
        

        self.start_lr = start_lr
        self.num_epochs = num_epochs

        self.log_interval = log_interval
        self.examples_per_log = examples_per_log
        self.eval_batches = eval_batches

        self.global_step = 0
        self.steps_per_epoch = len(self.train_loader) if self.train_loader is not None else 0

        

    def validate_loader(
        self,
        loader_eval,
        epoch,
        prefix = "val",
        max_examples = 3,
        sr = 16000,
        pad_id = None,
    ):
        pad_id = self.pad_id if pad_id is None else pad_id

        self.encoder.eval()
        self.decoder.eval()

        total_word_errors = 0
        total_words = 0
        total_char_errors = 0
        total_chars = 0
        n_batches = 0
        n_samples = 0

        all_pred_texts = []
        all_ref_texts = []

        with torch.no_grad():
            for batch in tqdm(loader_eval, desc=f"Evaluating {prefix}"):
                n_batches += 1
                if n_batches > self.eval_batches:
                    break

                audio = batch['audio'].to(self.device)
                texts = batch['text'].to(self.device)
                audio_lengths = batch['audio_lengths'].to(self.device)
                sr_batch = batch.get('sr', sr)

                mels, mel_lens = compute_log_melspectrogram(audio, audio_lengths, sr=sr_batch, device=self.device, augment=False)
                mels = mels.transpose(2, 1)

                padded_mel_len = mels.size(1)
                conv1_padded = compute_downsampled_len(torch.tensor(padded_mel_len, device=self.device))
                max_enc_len = compute_downsampled_len(conv1_padded)
                conv1_lens = compute_downsampled_len(mel_lens)
                enc_lens = compute_downsampled_len(conv1_lens)

                B = audio.size(0)
                enc_key_padding_mask = torch.ones(B, max_enc_len, device=self.device, dtype=torch.bool)
                for i in range(B):
                    enc_key_padding_mask[i, :enc_lens[i]] = False

                enc_out = self.encoder(mels.unsqueeze(1), key_padding_mask=enc_key_padding_mask)

                pred_ids = self.decoder.autoregressive_decode(
                    enc_out,
                    max_length=200,
                    bos_id=self.bos_id,
                    eos_id=self.eos_id,
                    notimestamps_id=self.notimestamps_id,
                    cross_key_padding_mask=enc_key_padding_mask,
                    device=self.device
                )

                if pred_ids.size(1) > 2:
                    pred_ids = pred_ids[:, 2:]
                else:
                    pred_ids = pred_ids[:, 1:]

                tgt_ids = texts[:, 1:].cpu()

                decoder_inputs = texts[:, :-1]
                targets = texts[:, 1:]
                decoder_self_key_padding_mask = (decoder_inputs == pad_id)
                logits = self.decoder(
                    decoder_inputs,
                    enc_out,
                    self_key_padding_mask=decoder_self_key_padding_mask,
                    cross_key_padding_mask=enc_key_padding_mask
                )
    

                for i in range(pred_ids.size(0)):
                    pred_seq = pred_ids[i]
                    tgt_seq = tgt_ids[i]

                    pred_text = tokens_to_text(pred_seq, self.tokenizer, pad_id=pad_id)
                    ref_text = tokens_to_text(tgt_seq, self.tokenizer, pad_id=pad_id)

                    total_char_errors += levenshtein(list(ref_text), list(pred_text))
                    total_chars += max(1, len(ref_text))

                    ref_words = ref_text.split()
                    pred_words = pred_text.split()
                    total_word_errors += levenshtein(ref_words, pred_words)
                    total_words += max(1, len(ref_words))

                    all_pred_texts.append(pred_text)
                    all_ref_texts.append(ref_text)
                    n_samples += 1

        wer = total_word_errors / total_words 
        cer = total_char_errors / total_chars 
    

        table = wandb.Table(columns=["target", "prediction", "CER", "WER"])
        indices = list(range(n_samples))
        random.shuffle(indices)
        for idx in indices[:max_examples]:
            pred_text = all_pred_texts[idx].strip().lower()
            tgt_text = all_ref_texts[idx].strip().lower()

            cer_val = levenshtein(list(tgt_text), list(pred_text)) / max(1, len(tgt_text))
            wer_val = levenshtein(tgt_text.split(), pred_text.split()) / max(1, len(tgt_text.split()))

            table.add_data(tgt_text, pred_text, float(cer_val), float(wer_val))

        wandb.log({f"{prefix}/examples_epoch_{epoch}": table})

        return wer, cer, n_samples

    def train(self, start_epoch = 0):
        epoch = start_epoch
        self.global_step = len(self.train_loader) * start_epoch

        for epoch in range(start_epoch, self.num_epochs):
            self.encoder.train()
            self.decoder.train()

            total_loss = 0.0
            total_word_errors = 0
            total_words = 0
            total_char_errors = 0
            total_chars = 0
            j = -1

            for batch in tqdm(self.train_loader, desc=f"train epoch {epoch}"):
                j += 1

                if epoch > 0:
                    epoch_step = epoch - 6
                else:
                    epoch_step = epoch

                lr = compute_lr_for_step(self.global_step, epoch_step, steps_per_epoch=self.steps_per_epoch)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

                audio = batch['audio'].to(self.device)
                texts = batch['text'].to(self.device)
                audio_lengths = batch['audio_lengths'].to(self.device)
                sr_batch = batch.get('sr', 16000)

                mels, mel_lens = compute_log_melspectrogram(audio, audio_lengths, sr=sr_batch, device=self.device, augment=True)
                mels = mels.transpose(2, 1)

                padded_mel_len = mels.size(1)
                conv1_padded = compute_downsampled_len(torch.tensor(padded_mel_len, device=self.device))
                max_enc_len = compute_downsampled_len(conv1_padded)
                conv1_lens = compute_downsampled_len(mel_lens)
                enc_lens = compute_downsampled_len(conv1_lens)

                B = audio.size(0)
                enc_key_padding_mask = torch.ones(B, max_enc_len, device=self.device, dtype=torch.bool)
                for i in range(B):
                    enc_key_padding_mask[i, :enc_lens[i]] = False

                enc_out = self.encoder(mels.unsqueeze(1), key_padding_mask=enc_key_padding_mask)

                decoder_inputs = texts[:, :-1]
                targets = texts[:, 1:]

                logits = self.decoder(
                    decoder_inputs,
                    enc_out,
                    self_key_padding_mask=(decoder_inputs == self.pad_id),
                    cross_key_padding_mask=enc_key_padding_mask
                )

                loss = self.loss_fn(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
                total_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()

                grad_norm = compute_grad_norm(list(self.encoder.parameters()) + list(self.decoder.parameters()))
                self.optimizer.step()
                self.global_step += 1

                if j % self.log_interval == 0:
                    with torch.no_grad():
                        probs = F.softmax(logits, dim=-1)
                        max_probs, pred_ids = probs.max(dim=-1)
                        threshold = 0.33
                        low_conf_mask = max_probs < threshold
                        pred_ids_masked = pred_ids.masked_fill(low_conf_mask, self.pad_id)

                        pred_list = [p.tolist() for p in pred_ids_masked.cpu()]
                        tgt_list = [t.tolist() for t in targets.cpu()]

                        pred_texts = self.tokenizer.batch_decode(pred_list, skip_special_tokens=True)
                        tgt_texts = self.tokenizer.batch_decode(tgt_list, skip_special_tokens=True)

                        table = wandb.Table(columns=["target", "prediction", "CER", "WER"])
                        for i_sample, (pred_text, tgt_text) in enumerate(zip(pred_texts, tgt_texts)):
                            pred_text = pred_text.strip().lower()
                            tgt_text = tgt_text.strip().lower()

                            cer_val = levenshtein(list(tgt_text), list(pred_text))
                            cer_val = cer_val / max(1, len(tgt_text))

                            wer_val = levenshtein(tgt_text.split(), pred_text.split())
                            wer_val = wer_val / max(1, max(1, len(tgt_text.split())))

                            total_char_errors += levenshtein(list(tgt_text), list(pred_text))
                            total_chars += max(1, len(tgt_text))

                            total_word_errors += levenshtein(tgt_text.split(), pred_text.split())
                            total_words += max(1, len(tgt_text.split()))

                            if i_sample < self.examples_per_log:
                                table.add_data(tgt_text, pred_text, float(cer_val), float(wer_val))

                        wandb.log({
                            "train/loss_batch": loss.item(),
                            "train/grad_norm": grad_norm,
                            "train/lr": lr,
                            "train/running_loss": total_loss / (j + 1),
                            "train/running_WER": (total_word_errors / total_words) if total_words > 0 else 0.0,
                            "train/running_CER": (total_char_errors / total_chars) if total_chars > 0 else 0.0,
                            f"train/examples_epoch{epoch}": table,
                            "epoch": epoch,
                            "step": epoch * 10**6 + j
                        })
                    gc.collect()
                    torch.cuda.empty_cache()

                if j % self.log_interval == 0:
                    cur_wer = (total_word_errors / total_words) if total_words > 0 else 0.0
                    cur_cer = (total_char_errors / total_chars) if total_chars > 0 else 0.0
                    print(f"Batch {j} Loss: {loss.item():.4f} | running WER: {cur_wer:.3f} CER: {cur_cer:.3f} grad_norm: {grad_norm:.3f} lr: {lr:.6f}")

            val_clean_wer, val_clean_cer, n_clean = self.validate_loader(
                self.loader_val_clean, epoch, prefix="val_clean", max_examples=2, sr=sr_batch, pad_id=self.pad_id
            )
            val_other_wer, val_other_cer, n_other = self.validate_loader(
                self.loader_val_other, epoch, prefix="val_other", max_examples=2, sr=sr_batch, pad_id=self.pad_id
            )

            wandb.log({
                "val_clean/wer": val_clean_wer,
                "val_clean/cer": val_clean_cer,
                "val_other/wer": val_other_wer,
                "val_other/cer": val_other_cer,
                "epoch": epoch
            })

            
            k = np.random.randint(0, audio.size(0))
            mel_for_plot = mels[k].transpose(0, 1)
            fig = plot_mel_to_image(mel_for_plot, title=f"train_mel_epoch{epoch}")
            wandb.log({f"train/mel_epoch{epoch}": wandb.Image(fig)})
            plt.close(fig)
            length = int(audio_lengths[k].item())
            audio_np = audio[k, :length].cpu().numpy()
            wandb.log({f"train/audio_epoch{epoch}": wandb.Audio(audio_np, sample_rate=sr_batch)})
            

            print(f"Epoch {epoch+1} VAL_CLEAN WER: {val_clean_wer:.4f} CER: {val_clean_cer:.4f} (n={n_clean})")
            print(f"Epoch {epoch+1} VAL_OTHER WER: {val_other_wer:.4f} CER: {val_other_cer:.4f} (n={n_other})")

            self.save_models(encoder_path=self.encoder_save_path, decoder_path=self.decoder_save_path)

        try:
            wandb.finish()
        except Exception:
            pass

    def save_models(self, encoder_path, decoder_path):
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def load_models(self, encoder_path, decoder_path, map_location = 'cuda'):
        map_location = map_location or self.device
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
        self.optimizer = Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.start_lr)
