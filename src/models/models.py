import torch
from torch import nn
import torch.nn.functional as F
import math


def get_sinusoidal_positional_encoding(seq_len, dim):
    pe = torch.zeros(seq_len, dim)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=208, num_heads=8):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, key_padding_mask=None):
        attn_output, _ = self.mhsa(x, x, x, key_padding_mask=key_padding_mask)
        x = self.ln1(x + attn_output)
        ff_output = self.ff(x)
        x = self.ln2(x + ff_output)
        return x

class Encoder(nn.Module):
    def __init__(self, embed_dim=208, num_heads=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 5), stride=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 5), stride=2)
        self.bn2 = nn.BatchNorm2d(16)

        self.transformer_block1 = TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads)
        self.transformer_block2 = TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads)
        self.transformer_block3 = TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, x, key_padding_mask=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), x.size(1), -1) 
        seq_len = x.size(1)
        embed_dim = x.size(2)
        pos_enc = get_sinusoidal_positional_encoding(seq_len, embed_dim).to(x.device).unsqueeze(0)
        x = x + pos_enc

        x = self.transformer_block1(x, key_padding_mask=key_padding_mask)
        x = self.transformer_block2(x, key_padding_mask=key_padding_mask)
        x = self.transformer_block3(x, key_padding_mask=key_padding_mask)

        return x
    
class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim=208, num_heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ln3 = nn.LayerNorm(embed_dim)

    def forward(self, x, enc_out, self_mask=None, self_key_padding_mask=None, cross_key_padding_mask=None):
        self_attn_output, _ = self.self_attn(
            query=x, key=x, value=x,
            attn_mask=self_mask,
            key_padding_mask=self_key_padding_mask
        )
        x = self.ln1(x + self_attn_output)

        cross_attn_output, _ = self.cross_attn(
            query=x, key=enc_out, value=enc_out,
            key_padding_mask=cross_key_padding_mask
        )
        x = self.ln2(x + cross_attn_output)

        ff_output = self.ff(x)
        x = self.ln3(x + ff_output)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size=51865, embed_dim=208, num_heads=8, num_layers=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(embed_dim=embed_dim, num_heads=num_heads) for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, tokens, enc_out, self_key_padding_mask=None, cross_key_padding_mask=None):
        B, S = tokens.shape
        x = self.embedding(tokens)

        pos_enc = get_sinusoidal_positional_encoding(S, self.embed_dim).to(x.device).unsqueeze(0) 
        x = x + pos_enc

        causal_mask = torch.triu(torch.full((S, S), float('-inf'), dtype=torch.float, device=x.device), diagonal=1)

        for layer in self.layers:
            x = layer(
                x, enc_out,
                self_mask=causal_mask,
                self_key_padding_mask=self_key_padding_mask,
                cross_key_padding_mask=cross_key_padding_mask
            )

        x = self.final_ln(x)
        logits = self.lm_head(x) 
        return logits

    def autoregressive_decode(
        self,
        enc_out,
        max_length=200,
        bos_id=50258,
        eos_id=50257,
        notimestamps_id=50363,
        pad_id=50256,
        cross_key_padding_mask=None,
        device=None,
        eos_threshold=0.6,
        recent_max_p_threshold=0.25,
        recent_window=5,
        min_len=3,
    ):
        if device is None:
            device = enc_out.device

        B = enc_out.size(0)
        prompt = torch.tensor([[bos_id, notimestamps_id]], device=device).repeat(B, 1)
        generated = prompt  

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        recent_max_history = [[] for _ in range(B)]

        for step in range(2, max_length):
            x = self.embedding(generated) 
            pos_enc = get_sinusoidal_positional_encoding(generated.size(1), self.embed_dim).to(device).unsqueeze(0)
            x = x + pos_enc

            current_len = generated.size(1)
            causal_mask = torch.triu(torch.full((current_len, current_len), float('-inf'), device=device), diagonal=1)

            self_key_padding_mask = (generated == pad_id) if pad_id is not None else None

            x_layer = x
            for layer in self.layers:
                x_layer = layer(
                    x_layer, enc_out,
                    self_mask=causal_mask,
                    self_key_padding_mask=self_key_padding_mask,
                    cross_key_padding_mask=cross_key_padding_mask
                )

            x_out = self.final_ln(x_layer)
            logits = self.lm_head(x_out[:, -1, :])  

            probs = F.softmax(logits, dim=-1)

            p_eos = probs[:, eos_id] if eos_id is not None else torch.zeros(B, device=device)
            max_p_vals, _ = torch.max(probs, dim=-1)
            for i in range(B):
                recent_max_history[i].append(float(max_p_vals[i].item()))
                if len(recent_max_history[i]) > recent_window:
                    recent_max_history[i].pop(0)

            recent_max = torch.tensor(
                [max(h[-recent_window:]) if len(h) > 0 else 0.0 for h in recent_max_history],
                device=device
            )


            cond_eos = p_eos > eos_threshold 

            if step > min_len:
                cond_unconf = recent_max < recent_max_p_threshold
            else:
                cond_unconf = torch.zeros(B, dtype=torch.bool, device=device)

            should_finish = cond_eos | cond_unconf

            new_finished = (~finished) & should_finish 
            finished = finished | should_finish
            next_token = torch.argmax(probs, dim=-1).unsqueeze(1) 

            if eos_id is not None:
                next_token[finished] = eos_id
            else:
                next_token[finished] = pad_id if pad_id is not None else next_token[finished]

            generated = torch.cat([generated, next_token], dim=1) 

            if torch.all(finished):
                break

        return generated 


    def beam_search_decode(
        self,
        enc_out,
        beam_size: int = 4,
        max_length: int = 200,
        bos_id: int = 50258,
        eos_id: int = 50257,
        notimestamps_id: int = 50363,
        pad_id: int = 50256,
        length_penalty_alpha: float = 0.6, 
        n_best: int = 1,
        device: torch.device = None,
        cross_key_padding_mask = None
    ):
        if device is None:
            device = enc_out.device
        B = enc_out.size(0)
        vocab_size = self.lm_head.out_features

        def _forward_tokens(tokens_batch, enc_batch, cross_key_padding_mask_batch=None):
            x = self.embedding(tokens_batch)
            pos_enc = get_sinusoidal_positional_encoding(tokens_batch.size(1), self.embed_dim).to(device).unsqueeze(0)
            x = x + pos_enc
            current_len = tokens_batch.size(1)
            causal_mask = torch.triu(torch.full((current_len, current_len), float('-inf'), device=device), diagonal=1)
            self_key_padding_mask = (tokens_batch == pad_id) if pad_id is not None else None
            x_layer = x
            for layer in self.layers:
                x_layer = layer(
                    x_layer, enc_batch,
                    self_mask=causal_mask,
                    self_key_padding_mask=self_key_padding_mask,
                    cross_key_padding_mask=cross_key_padding_mask_batch
                )
            x_out = self.final_ln(x_layer)
            logits = self.lm_head(x_out)
            return logits

        all_results = []
        def length_norm(raw_score, seq_len, alpha):
            if alpha == 0.0:
                return float(raw_score)
            lp = ((5.0 + seq_len) ** alpha) / ((5.0 + 1.0) ** alpha)
            return float(raw_score) / lp

        for b in range(B):
            enc_b = enc_out[b:b+1]
            prompt = torch.tensor([[bos_id, notimestamps_id]], device=device)
            beams = [prompt.clone() for _ in range(beam_size)]
            beam_raw_scores = torch.tensor([0.0] + [-1e9] * (beam_size - 1), device=device) 
            finished_hyps = []  

            for step in range(2, max_length):
                max_len = max(b_seq.size(1) for b_seq in beams)
                padded_beams = [F.pad(b_seq, (0, max_len - b_seq.size(1)), value=pad_id) for b_seq in beams]
                tokens_stack = torch.cat(padded_beams, dim=0) 
                enc_repeat = enc_b.repeat(len(beams), 1, 1)
                logits = _forward_tokens(tokens_stack, enc_repeat, cross_key_padding_mask_batch=cross_key_padding_mask)
                last_logits = logits[:, -1, :] 
                log_probs = F.log_softmax(last_logits, dim=-1) 

                for i_beam, beam_seq in enumerate(beams):
                    if beam_seq[0, -1].item() == eos_id:
                        mask = torch.full((vocab_size,), -1e9, device=device)
                        mask[eos_id] = 0.0
                        log_probs[i_beam] = mask

                cand_scores = beam_raw_scores.unsqueeze(1) + log_probs 
                flat_scores = cand_scores.view(-1)
                k = min(beam_size, flat_scores.size(0))
                topk_scores, topk_indices = torch.topk(flat_scores, k=k)

                new_beams = []
                new_beam_raw_scores = []
                for sc, idx in zip(topk_scores.tolist(), topk_indices.tolist()):
                    beam_idx = idx // vocab_size
                    token_id = idx % vocab_size
                    seq = beams[beam_idx].clone()
                    seq = torch.cat([seq, torch.tensor([[token_id]], device=device)], dim=1)
                    new_beams.append(seq)
                    new_beam_raw_scores.append(float(sc))

                beams = new_beams
                beam_raw_scores = torch.tensor(new_beam_raw_scores, device=device)

                still_alive = []
                still_alive_scores = []
                for i_beam, b_seq in enumerate(beams):
                    last_tok = b_seq[0, -1].item()
                    if last_tok == eos_id:
                        finished_hyps.append((b_seq.clone(), float(beam_raw_scores[i_beam].item())))
                    else:
                        still_alive.append(b_seq)
                        still_alive_scores.append(float(beam_raw_scores[i_beam].item()))

                if len(still_alive) == 0:
                    break

                num_needed = beam_size - len(still_alive)
                beams = still_alive.copy()
                beam_raw_scores = torch.tensor(still_alive_scores, device=device) if len(still_alive_scores) > 0 else torch.tensor([], device=device)

                if num_needed > 0 and len(finished_hyps) > 0:
                    finished_pool = sorted(finished_hyps, key=lambda x: x[1], reverse=True)
                    for (s, raw_sc) in finished_pool[:num_needed]:
                        beams.append(s.clone())
                        beam_raw_scores = torch.cat([beam_raw_scores, torch.tensor([raw_sc], device=device)]) if beam_raw_scores.numel() > 0 else torch.tensor([raw_sc], device=device)

                while len(beams) < beam_size:
                    beams.append(beams[0].clone())
                    beam_raw_scores = torch.cat([beam_raw_scores, beam_raw_scores[:1]]) if beam_raw_scores.numel() > 0 else torch.tensor([0.0], device=device)

            final_pool = []
            for (s, raw_sc) in finished_hyps:
                seq_len = s.size(1)
                norm_sc = length_norm(raw_sc, seq_len, length_penalty_alpha)
                final_pool.append((s, norm_sc))
            if len(final_pool) == 0:
                for i_seq, s in enumerate(beams):
                    seq_len = s.size(1)
                    raw_sc = float(beam_raw_scores[i_seq].item())
                    norm_sc = length_norm(raw_sc, seq_len, length_penalty_alpha)
                    final_pool.append((s, norm_sc))

            final_pool.sort(key=lambda x: x[1], reverse=True)
            best_hyps = final_pool[:n_best]
            best_seqs = [s for (s, sc) in best_hyps]

            max_len = max(s.size(1) for s in best_seqs)
            padded = []
            for s in best_seqs:
                pad_len = max_len - s.size(1)
                if pad_len > 0:
                    pad_tensor = torch.full((1, pad_len), pad_id, dtype=torch.long, device=device)
                    padded_seq = torch.cat([s, pad_tensor], dim=1)
                else:
                    padded_seq = s
                padded.append(padded_seq)
            stacked = torch.cat(padded, dim=0)  
            all_results.append(stacked)

        max_len_overall = max(r.size(1) for r in all_results)
        out_list = []
        for r in all_results:
            if r.size(1) < max_len_overall:
                pad_len = max_len_overall - r.size(1)
                pad_tensor = torch.full((r.size(0), pad_len), pad_id, dtype=torch.long, device=device)
                r_padded = torch.cat([r, pad_tensor], dim=1)
            else:
                r_padded = r
            out_list.append(r_padded.unsqueeze(0))
        out_tensor = torch.cat(out_list, dim=0) 
        return out_tensor   