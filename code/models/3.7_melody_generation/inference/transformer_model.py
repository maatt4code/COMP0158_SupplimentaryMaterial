"""The melody transformer's architecture and vocabulary. Definition only.

Runtime code. `../train/train_transformer.py` trains this; the conductor loads
it. The class lives on the inference side so the runtime never imports a
training script to rebuild an architecture before loading a checkpoint.

The vocabulary is deliberately tiny: pitch OFFSETS rather than absolute pitches,
a small ladder of duration ratios, and a rest token. Offsets make transposition
free, which is what lets one model serve every register the conductor asks for.

Conditioning is a continuous (valence, arousal) vector projected to the model
width and added to every token embedding, so the same weights generate
differently across the affect plane without a separate model per mood.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Vocabulary mapping: Pitch offsets (-12 to +12) -> 25 tokens, Duration ratios -> 8 tokens, Rest -> 1 token
PITCH_RANGE = list(range(-12, 13))
DUR_RANGES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
TOKEN_REST = 0
TOKEN_PAD = 1
TOKEN_EOP = 34

# Vocabulary mapping
pitch_to_tok = {p: i + 2 for i, p in enumerate(PITCH_RANGE)}
tok_to_pitch = {i + 2: p for i, p in enumerate(PITCH_RANGE)}
VOCAB_SIZE = len(PITCH_RANGE) + 10



class MicroMelodicTransformer(nn.Module):
    """Causal Micro-Transformer for real-time CPU generation with configurable depth."""
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=128, n_heads=4, n_layers=3, d_ffn=512):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ffn = d_ffn
        self.vocab_size = vocab_size
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 64, d_model))
        self.cond_proj = nn.Linear(2, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ffn,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, va_cond):
        b, t = x.size()
        tok_embeds = self.tok_emb(x) + self.pos_emb[:, :t, :]
        va_embeds = self.cond_proj(va_cond).unsqueeze(1)
        h = tok_embeds + va_embeds
        
        # Causal mask for autoregressive generation
        mask = torch.triu(torch.full((t, t), float('-inf')), diagonal=1).to(x.device)
        out = self.transformer(h, mask=mask)
        logits = self.head(out)
        return logits

    @torch.no_grad()
    def generate(self, va_cond, max_len=16, temperature=0.9, top_p=0.85):
        self.eval()
        device = next(self.parameters()).device
        if va_cond.dim() == 1:
            va_cond = va_cond.unsqueeze(0)
        va_cond = va_cond.to(device)
        
        curr_seq = [pitch_to_tok[0]] # Start at tonic root 0
        for _ in range(max_len):
            x_in = torch.tensor([curr_seq], dtype=torch.long, device=device)
            logits = self.forward(x_in, va_cond)[:, -1, :] # [1, VOCAB_SIZE]
            logits = logits / max(temperature, 1e-5)
            
            # Top-p (nucleus) sampling
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            probs[indices_to_remove] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            next_tok = torch.multinomial(probs, num_samples=1).item()
            if next_tok in [TOKEN_REST, TOKEN_PAD, TOKEN_EOP] or len(curr_seq) >= max_len:
                break
            curr_seq.append(next_tok)
        return curr_seq


