"""
PROJECT AM v10 — Tiny Recursive Model (TRM)
Based on "Less is More: Recursive Reasoning with Tiny Networks"
"""

import math
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from am import TRMConfig, PAD_TOKEN, NUM_TOKENS

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if not HAS_TORCH:
    class TRM:
        """Stub — install PyTorch: pip install torch"""
        def __init__(self, config):
            raise ImportError("PyTorch required for TRM. Run: pip install torch")

    class ARCTRMDataset:
        def __init__(self, *a, **kw):
            raise ImportError("PyTorch required. Run: pip install torch")

    class EMA:
        def __init__(self, *a, **kw):
            raise ImportError("PyTorch required. Run: pip install torch")

else:
    class TransformerBlock(nn.Module):
        def __init__(self, dim, num_heads, mlp_ratio, dropout):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
            h = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim), nn.Dropout(dropout))

        def forward(self, x):
            h = self.norm1(x)
            h, _ = self.attn(h, h, h)
            x = x + h
            h = self.norm2(x)
            x = x + self.mlp(h)
            return x

    class TRM(nn.Module):
        """
        Tiny Recursive Model — 2-layer network applied recursively H*L times.
        Deep supervision at every H step. ~7M params at dim=256.
        """
        def __init__(self, config: TRMConfig):
            super().__init__()
            self.config = config
            d = config.dim
            self.token_emb = nn.Embedding(NUM_TOKENS, d)
            self.pos_emb = nn.Parameter(torch.randn(1, config.max_x_len + config.max_y_len * 2, d) * 0.02)
            self.layers = nn.ModuleList([
                TransformerBlock(d, config.num_heads, config.mlp_ratio, config.dropout)
                for _ in range(config.num_layers)
            ])
            self.answer_gate = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, d))
            self.output_norm = nn.LayerNorm(d)
            self.output_head = nn.Linear(d, NUM_TOKENS)
            self.apply(self._init_weights)
            n = sum(p.numel() for p in self.parameters())
            print(f"  TRM initialized: {n:,} parameters")

        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        def forward(self, x_tokens, y_tokens, targets=None):
            B, Lx = x_tokens.shape
            Ly = y_tokens.shape[1]
            x = self.token_emb(x_tokens) + self.pos_emb[:, :Lx, :]
            y = self.token_emb(y_tokens) + self.pos_emb[:, Lx:Lx + Ly, :]
            z = torch.zeros_like(y)
            all_logits, all_losses = [], []
            for t in range(self.config.H_cycles):
                for n in range(self.config.L_cycles):
                    combined = torch.cat([x, y, z], dim=1)
                    for layer in self.layers:
                        combined = layer(combined)
                    z = combined[:, Lx + Ly:, :]
                y = self.answer_gate(torch.cat([y, z], dim=-1))
                logits = self.output_head(self.output_norm(y))
                all_logits.append(logits)
                if targets is not None:
                    loss = F.cross_entropy(logits.reshape(-1, NUM_TOKENS), targets.reshape(-1), ignore_index=PAD_TOKEN)
                    all_losses.append(loss)
            result = {"logits": all_logits[-1], "all_logits": all_logits}
            if targets is not None:
                result["total_loss"] = sum(all_losses) / len(all_losses)
                result["losses"] = all_losses
            return result

        @torch.no_grad()
        def predict(self, x_tokens, y_init):
            self.eval()
            return self.forward(x_tokens, y_init)["logits"].argmax(dim=-1)

    class ARCTRMDataset(Dataset):
        def __init__(self, encoded_tasks, max_x=1024, max_y=900):
            self.tasks = encoded_tasks
            self.max_x = max_x
            self.max_y = max_y

        def __len__(self):
            return len(self.tasks)

        def __getitem__(self, idx):
            t = self.tasks[idx]
            x, y = t["x"], t["y_target"]
            if len(x) > self.max_x:
                x = x[:self.max_x]
            else:
                x = np.pad(x, (0, self.max_x - len(x)), constant_values=PAD_TOKEN)
            if len(y) > self.max_y:
                y = y[:self.max_y]
            else:
                y = np.pad(y, (0, self.max_y - len(y)), constant_values=PAD_TOKEN)
            y_init = np.full_like(y, 0)
            return {
                "x": torch.tensor(x, dtype=torch.long),
                "y_init": torch.tensor(y_init, dtype=torch.long),
                "y_target": torch.tensor(y, dtype=torch.long),
            }

    class EMA:
        def __init__(self, model, decay=0.999):
            self.decay = decay
            self.shadow = {name: p.clone().detach() for name, p in model.named_parameters()}

        def update(self, model):
            with torch.no_grad():
                for name, p in model.named_parameters():
                    self.shadow[name].mul_(self.decay).add_(p, alpha=1 - self.decay)

        def apply(self, model):
            self._backup = {name: p.clone() for name, p in model.named_parameters()}
            with torch.no_grad():
                for name, p in model.named_parameters():
                    p.copy_(self.shadow[name])

        def restore(self, model):
            with torch.no_grad():
                for name, p in model.named_parameters():
                    p.copy_(self._backup[name])
