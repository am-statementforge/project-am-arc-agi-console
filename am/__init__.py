"""
PROJECT AM v10 — Configuration
All hyperparameters in one place. Change here, affects everywhere.
"""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"

# ARC constants
MAX_GRID_SIZE = 30
NUM_COLORS = 10
PAD_TOKEN = 10
SEP_TOKEN = 11
NUM_TOKENS = 12  # 0-9 colors + pad + separator
MAX_SEQ_LEN = 2 * MAX_GRID_SIZE * MAX_GRID_SIZE + 10


@dataclass
class TRMConfig:
    """TRM hyperparameters — matches the paper's best settings."""
    dim: int = 256
    num_layers: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    H_cycles: int = 3       # deep supervision steps
    L_cycles: int = 6       # recursion steps per supervision step
    lr: float = 1e-4
    ttt_lr: float = 1e-4    # test-time training learning rate
    weight_decay: float = 0.01
    ema_decay: float = 0.999
    batch_size: int = 4
    epochs: int = 3000
    ttt_steps: int = 200    # test-time training steps per puzzle
    num_attempts: int = 2   # pass@2 for ARC evaluation
    max_x_len: int = 1024   # max input sequence length
    max_y_len: int = 900    # max output sequence length

    @property
    def total_recursions(self) -> int:
        return self.H_cycles * self.L_cycles
