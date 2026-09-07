"""Train the causal micro-transformer over folk phrase contours.

Step 3 of Section 3.7. The architecture is deliberately small: it runs on CPU
inside the live conductor, so it is sized for latency rather than for the last
fraction of a nat.

Architecture:
  - 3-Layer Causal Transformer (~1.5M parameters)
  - Small vocabulary: Pitch offsets [-12..+12], Duration ratios [0.25..4.0], Rest Silence tokens.
  - Conditioning: Continuous (Valence, Arousal) vector passed to prefix embedding.
  - Training Target: Learns spatial motif repetition, phrase contours, and ambient rest gaps.
  - Output: Saves lightweight CPU checkpoint 'melodic_transformer_cpu.pt' (< 6 MB).

Usage:
  python train_melodic_transformer.py --epochs 10 --batch_size 16
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
from transformer_model import (MicroMelodicTransformer, PITCH_RANGE,   # noqa: E402
                               DUR_RANGES, TOKEN_REST, TOKEN_PAD, TOKEN_EOP,
                               pitch_to_tok, tok_to_pitch, VOCAB_SIZE)

HERE = Path(__file__).resolve().parent
SECTION = HERE.parent
# Training writes to the section's data/, never to weights/: weights/ holds the
# checkpoints that ship and reproduce the reported results.
DATA_DIR = SECTION / "data"
CKPT_DIR = SECTION / "data" / "checkpoints"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CUSTOM_DATASET_PATH = DATA_DIR / "ambient_melodic_dataset.json"
MODEL_DATA_PATH = SECTION / "weights" / "markov_order2.json"

class AmbientMelodicDataset(Dataset):
    """Dataset built from parsed Essen phrase contours and synth motifs."""
    def __init__(self, data_path=CUSTOM_DATASET_PATH, seq_len=32):
        self.seq_len = seq_len
        self.samples = []
        
        # 1. Load generated ambient_melodic_dataset.json if present
        if data_path.exists():
            with open(data_path, 'r') as f:
                data = json.load(f)
            sequences = data.get("sequences", [])
            for seq in sequences:
                toks = []
                for step in seq:
                    off = max(-12, min(12, int(step.get("pitch_offset", 0))))
                    toks.append(pitch_to_tok.get(off, pitch_to_tok[0]))
                if len(toks) >= 4:
                    self.samples.append(toks)
        # 2. Fallback to essen_model.json
        elif MODEL_DATA_PATH.exists():
            with open(MODEL_DATA_PATH, 'r') as f:
                data = json.load(f)
            phrase_bank = data.get("phrase_bank", [])
            for item in phrase_bank[:2000]:
                shape = item.get("shape", [])
                toks = []
                for step in shape:
                    if isinstance(step, (list, tuple)) and len(step) >= 2:
                        pitch_off = max(-12, min(12, int(step[0])))
                        toks.append(pitch_to_tok.get(pitch_off, pitch_to_tok[0]))
                if len(toks) >= 4:
                    self.samples.append(toks)
        
        if not self.samples:
            # Fallback synthetic training sequences
            for _ in range(500):
                seq = [pitch_to_tok[p] for p in [0, 2, 4, 7, 4, 2, 0, TOKEN_REST, 0, 2, 4, 7]]
                self.samples.append(seq)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        toks = self.samples[idx]
        if len(toks) < self.seq_len:
            toks_padded = toks + [TOKEN_PAD] * (self.seq_len - len(toks))
        else:
            toks_padded = toks[:self.seq_len]
        
        x = torch.tensor(toks_padded[:-1], dtype=torch.long)
        y = torch.tensor(toks_padded[1:], dtype=torch.long)
        
        # Compute realistic VA condition from phrase content:
        # Valence = ratio of major intervals (0, 2, 4, 7, 9) vs minor/dissonant intervals
        pitches = [tok_to_pitch.get(t, 0) for t in toks if t in tok_to_pitch]
        major_count = sum(1 for p in pitches if (p % 12) in [0, 2, 4, 7, 9])
        v_label = float(2.0 * (major_count / max(1, len(pitches))) - 1.0)
        # Arousal = pitch variance / activity range
        p_std = float(np.std(pitches)) if len(pitches) > 1 else 0.0
        a_label = float(min(1.0, max(-1.0, (p_std - 3.0) / 3.0)))
        
        va = torch.tensor([v_label, a_label], dtype=torch.float32)
        return x, y, va


def train(epochs=100, batch_size=32, lr=1e-3, patience=7, n_layers=6, d_model=256, n_heads=8, d_ffn=1024, use_scheduler=True):
    import matplotlib.pyplot as plt
    
    full_dataset = AmbientMelodicDataset()
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MicroMelodicTransformer(d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ffn=d_ffn).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)
    
    warmup_epochs = 3
    if use_scheduler:
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None
        
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Training Causal Micro-Transformer (L={n_layers}, d={d_model}, h={n_heads}, {param_count:,} params, Scheduler={use_scheduler}) on {device} ({train_size} train, {val_size} val) max {epochs} epochs (Patience={patience})...")
    
    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y, va in train_loader:
            x, y, va = x.to(device), y.to(device), va.to(device)
            optimizer.zero_grad()
            logits = model(x, va)
            loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(x)
            
        train_loss /= train_size
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()
        
        # Validation pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, va in val_loader:
                x, y, va = x.to(device), y.to(device), va.to(device)
                logits = model(x, va)
                loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
                val_loss += loss.item() * len(x)
        val_loss /= val_size
        
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        
        # Early Stopping logic
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
            status_str = f" [Best Val: {best_val_loss:.4f}]"
        else:
            patience_counter += 1
            status_str = f" [Patience: {patience_counter}/{patience}]"
            
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{status_str}")
        
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1} (Best Val Loss: {best_val_loss:.4f})")
            break
        
    # Save best checkpoint with architecture-specific filename
    tag = f"L{n_layers}_d{d_model}"
    out_pth = CKPT_DIR / f"melodic_transformer_{tag}.pt"
    cfg_pth = CKPT_DIR / f"melodic_transformer_{tag}.json"
    default_pth = CKPT_DIR / "melodic_transformer_cpu.pt"
    default_cfg = CKPT_DIR / "melodic_transformer_cpu.json"
    
    cfg = {
        "n_layers": n_layers,
        "d_model": d_model,
        "n_heads": n_heads,
        "d_ffn": d_ffn,
        "vocab_size": VOCAB_SIZE,
        "param_count": param_count,
        "best_val_loss": best_val_loss
    }
    with open(cfg_pth, "w") as f:
        json.dump(cfg, f, indent=2)
    with open(default_cfg, "w") as f:
        json.dump(cfg, f, indent=2)
        
    if best_model_state is not None:
        torch.save(best_model_state, out_pth)
        torch.save(best_model_state, default_pth)
        print(f"Saved best CPU-ready model checkpoint to {out_pth} & {default_pth}")
    else:
        torch.save(model.state_dict(), out_pth)
        torch.save(model.state_dict(), default_pth)
    
    # Save training history JSON
    hist_path = HERE / "data" / f"training_history_{tag}.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
        
    # Generate publication-quality Loss Curve Plot (without baseline line)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
    epochs_range = history["epoch"]
    ax.plot(epochs_range, history["train_loss"], 'o-', color='#3182ce', lw=2.2, label='Training Loss (Cross-Entropy)')
    ax.plot(epochs_range, history["val_loss"], 's--', color='#e53e3e', lw=2.2, label='Validation Loss (Cross-Entropy)')
    
    ax.set_title(f'Causal Micro-Transformer (L={n_layers}, d={d_model}) Learning Curve', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='semibold')
    ax.set_ylabel('Cross-Entropy Loss (Nats)', fontsize=11, fontweight='semibold')
    ax.set_xticks(epochs_range)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
    plt.tight_layout()
    
    # Save plot to report figures
    fig_dir = HERE.parents[2] / "report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf_out = fig_dir / f"diagram_melodic_transformer_loss_{tag}.pdf"
    png_out = fig_dir / f"diagram_melodic_transformer_loss_{tag}.png"
    default_pdf = fig_dir / "diagram_melodic_transformer_loss.pdf"
    default_png = fig_dir / "diagram_melodic_transformer_loss.png"
    
    fig.savefig(pdf_out, bbox_inches='tight')
    fig.savefig(png_out, bbox_inches='tight', dpi=300)
    fig.savefig(default_pdf, bbox_inches='tight')
    fig.savefig(default_png, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved clean loss plots to {pdf_out.name} and {default_pdf.name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MicroMelodicTransformer")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size",
                        type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--n-layers", "--n_layers", dest="n_layers", type=int, default=6, help="Number of Causal Transformer layers (default: 6)")
    parser.add_argument("--d-model", "--d_model", dest="d_model", type=int, default=256, help="Model hidden dimension (default: 256)")
    parser.add_argument("--n-heads", "--n_heads", dest="n_heads", type=int, default=8, help="Number of attention heads (default: 8)")
    parser.add_argument("--d-ffn", "--d_ffn", dest="d_ffn", type=int, default=1024, help="Feed-forward dimension (default: 1024)")
    parser.add_argument("--no_scheduler", action="store_true", help="Disable Cosine Annealing learning rate scheduler")
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ffn=args.d_ffn,
        use_scheduler=not args.no_scheduler
    )
