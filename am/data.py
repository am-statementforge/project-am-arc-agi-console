"""
PROJECT AM v10 — ARC Data Pipeline
Downloads ARC-AGI datasets, loads tasks, encodes grids, augments data.
"""

import json
import os
import random
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import requests
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable

from am import (
    DATA_DIR, MAX_GRID_SIZE, NUM_COLORS, PAD_TOKEN, SEP_TOKEN,
    NUM_TOKENS, MAX_SEQ_LEN,
)

# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ARCGrid:
    """A single ARC grid (2D array of colors 0-9)."""
    data: np.ndarray  # shape (H, W), dtype int

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    def to_list(self) -> List[List[int]]:
        return self.data.tolist()

    @classmethod
    def from_list(cls, grid_list: List[List[int]]) -> "ARCGrid":
        return cls(data=np.array(grid_list, dtype=np.int32))


@dataclass
class ARCPair:
    """An input-output pair."""
    input: ARCGrid
    output: ARCGrid


@dataclass
class ARCTask:
    """A complete ARC task with train and test pairs."""
    task_id: str
    train_pairs: List[ARCPair]
    test_pairs: List[ARCPair]


# ─────────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────────

ARC1_URL = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
ARC2_URL = "https://github.com/arcprize/ARC-AGI-2/archive/refs/heads/main.zip"


def download_arc_data(version: str = "arc1", force: bool = False):
    """Download ARC-AGI datasets from GitHub."""
    if version == "arc1":
        url = ARC1_URL
        dest = DATA_DIR / "arc-agi-1"
        inner_prefix = "ARC-AGI-master"
    elif version == "arc2":
        url = ARC2_URL
        dest = DATA_DIR / "arc-agi-2"
        inner_prefix = "ARC-AGI-2-main"
    else:
        raise ValueError(f"Unknown version: {version}. Use 'arc1' or 'arc2'.")

    if dest.exists() and not force:
        # Check if data actually has task files
        json_files = list(dest.rglob("*.json"))
        if len(json_files) > 10:
            print(f"  {version} data already exists at {dest} ({len(json_files)} files)")
            return dest
        # Directory exists but empty/incomplete — redownload
        shutil.rmtree(dest)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / f"{version}.zip"

    print(f"  Downloading {version} from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(zip_path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"  {version}") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"  Extracting to {dest}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)

    # Move from nested directory to clean path
    extracted = DATA_DIR / inner_prefix
    if extracted.exists():
        if dest.exists():
            shutil.rmtree(dest)
        extracted.rename(dest)

    zip_path.unlink()  # cleanup zip
    json_count = len(list(dest.rglob("*.json")))
    print(f"  Done! {json_count} files in {dest}")
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# Load Tasks
# ─────────────────────────────────────────────────────────────────────────────

def load_tasks_from_dir(directory: Path) -> List[ARCTask]:
    """Load all ARC tasks from a directory of JSON files."""
    tasks = []
    json_files = sorted(directory.glob("*.json"))
    for fp in json_files:
        task_id = fp.stem
        with open(fp) as f:
            raw = json.load(f)

        train_pairs = []
        for pair in raw.get("train", []):
            train_pairs.append(ARCPair(
                input=ARCGrid.from_list(pair["input"]),
                output=ARCGrid.from_list(pair["output"]),
            ))

        test_pairs = []
        for pair in raw.get("test", []):
            test_pairs.append(ARCPair(
                input=ARCGrid.from_list(pair["input"]),
                output=ARCGrid.from_list(pair.get("output", pair["input"])),
            ))

        tasks.append(ARCTask(
            task_id=task_id,
            train_pairs=train_pairs,
            test_pairs=test_pairs,
        ))
    return tasks


def load_arc_dataset(version: str = "arc1", split: str = "training") -> List[ARCTask]:
    """Load ARC dataset. Downloads if not present."""
    base = DATA_DIR / ("arc-agi-1" if version == "arc1" else "arc-agi-2")

    if not base.exists():
        download_arc_data(version)

    # ARC-AGI-1 structure: data/training/*.json, data/evaluation/*.json
    # ARC-AGI-2 structure: similar
    candidates = [
        base / "data" / split,
        base / split,
    ]
    for c in candidates:
        if c.exists() and list(c.glob("*.json")):
            tasks = load_tasks_from_dir(c)
            print(f"  Loaded {len(tasks)} tasks from {c}")
            return tasks

    # Try to find any directory with JSON files
    for d in base.rglob("*"):
        if d.is_dir() and split in d.name.lower():
            json_files = list(d.glob("*.json"))
            if json_files:
                tasks = load_tasks_from_dir(d)
                print(f"  Loaded {len(tasks)} tasks from {d}")
                return tasks

    raise FileNotFoundError(
        f"Could not find {split} data in {base}. "
        f"Directories found: {[str(d) for d in base.iterdir() if d.is_dir()]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Encode Grids → Token Sequences
# ─────────────────────────────────────────────────────────────────────────────

def encode_grid(grid: ARCGrid) -> List[int]:
    """
    Encode a grid as a flat token sequence.
    Format: [row0_col0, row0_col1, ..., SEP, row1_col0, ...]
    """
    tokens = []
    for r in range(grid.height):
        if r > 0:
            tokens.append(SEP_TOKEN)
        for c in range(grid.width):
            tokens.append(int(grid.data[r, c]))
    return tokens


def decode_tokens_to_grid(tokens: List[int], height: int, width: int) -> ARCGrid:
    """Decode token sequence back to a grid."""
    grid = np.full((height, width), 0, dtype=np.int32)
    r, c = 0, 0
    for t in tokens:
        if t == PAD_TOKEN:
            continue
        if t == SEP_TOKEN:
            r += 1
            c = 0
            if r >= height:
                break
            continue
        if t < NUM_COLORS and r < height and c < width:
            grid[r, c] = t
            c += 1
    return ARCGrid(data=grid)


def encode_task_for_trm(task: ARCTask, pair_idx: int = -1) -> Dict[str, np.ndarray]:
    """
    Encode an ARC task for TRM input.

    x_tokens = [demo1_input SEP demo1_output SEP demo2_input SEP demo2_output SEP ... test_input]
    y_target = [test_output tokens]

    If pair_idx == -1, use last test pair. Otherwise use specified index.
    """
    x_parts = []

    # Encode all training demonstrations
    for pair in task.train_pairs:
        x_parts.extend(encode_grid(pair.input))
        x_parts.append(SEP_TOKEN)
        x_parts.extend(encode_grid(pair.output))
        x_parts.append(SEP_TOKEN)

    # Encode test input
    test_idx = pair_idx if pair_idx >= 0 else 0
    if test_idx < len(task.test_pairs):
        test_pair = task.test_pairs[test_idx]
    else:
        test_pair = task.test_pairs[0]

    x_parts.extend(encode_grid(test_pair.input))

    # Target is the test output
    y_tokens = encode_grid(test_pair.output)

    return {
        "x": np.array(x_parts, dtype=np.int64),
        "y_target": np.array(y_tokens, dtype=np.int64),
        "height": test_pair.output.height,
        "width": test_pair.output.width,
        "task_id": task.task_id,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment_grid(grid: ARCGrid, aug_type: str) -> ARCGrid:
    """Apply augmentation to a grid."""
    d = grid.data.copy()
    if aug_type == "rot90":
        d = np.rot90(d, 1)
    elif aug_type == "rot180":
        d = np.rot90(d, 2)
    elif aug_type == "rot270":
        d = np.rot90(d, 3)
    elif aug_type == "fliph":
        d = np.fliplr(d)
    elif aug_type == "flipv":
        d = np.flipud(d)
    elif aug_type == "transpose":
        d = d.T
    elif aug_type == "color_perm":
        # Random permutation of colors 1-9 (keep 0 as background)
        perm = list(range(10))
        subset = perm[1:]
        random.shuffle(subset)
        perm[1:] = subset
        d = np.vectorize(lambda x: perm[x])(d)
    return ARCGrid(data=d.astype(np.int32))


def augment_task(task: ARCTask, aug_type: str) -> ARCTask:
    """Apply the same augmentation to all grids in a task."""
    new_train = [
        ARCPair(
            input=augment_grid(p.input, aug_type),
            output=augment_grid(p.output, aug_type),
        ) for p in task.train_pairs
    ]
    new_test = [
        ARCPair(
            input=augment_grid(p.input, aug_type),
            output=augment_grid(p.output, aug_type),
        ) for p in task.test_pairs
    ]
    return ARCTask(
        task_id=f"{task.task_id}_{aug_type}",
        train_pairs=new_train,
        test_pairs=new_test,
    )


AUG_TYPES = ["rot90", "rot180", "rot270", "fliph", "flipv", "transpose", "color_perm"]


def augment_dataset(tasks: List[ARCTask], n_augments: int = 3) -> List[ARCTask]:
    """Augment each task with random augmentations."""
    augmented = list(tasks)  # keep originals
    for task in tasks:
        chosen = random.sample(AUG_TYPES, min(n_augments, len(AUG_TYPES)))
        for aug in chosen:
            augmented.append(augment_task(task, aug))
    return augmented


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Task Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_tasks(n: int = 100) -> List[ARCTask]:
    """Generate simple synthetic ARC-like tasks for pretraining."""
    tasks = []
    generators = [
        _gen_mirror_task,
        _gen_rotate_task,
        _gen_fill_task,
        _gen_tile_task,
        _gen_color_swap_task,
    ]
    for i in range(n):
        gen = random.choice(generators)
        task = gen(f"synth_{i:04d}")
        if task is not None:
            tasks.append(task)
    return tasks


def _random_grid(h: int, w: int, n_colors: int = 4) -> ARCGrid:
    """Generate a random grid."""
    colors = random.sample(range(10), min(n_colors, 10))
    data = np.random.choice(colors, size=(h, w)).astype(np.int32)
    return ARCGrid(data=data)


def _gen_mirror_task(task_id: str) -> ARCTask:
    """Generate a horizontal mirror task."""
    pairs = []
    for _ in range(random.randint(3, 5)):
        h, w = random.randint(2, 8), random.randint(2, 8)
        inp = _random_grid(h, w)
        out = ARCGrid(data=np.fliplr(inp.data))
        pairs.append(ARCPair(input=inp, output=out))
    return ARCTask(task_id=task_id, train_pairs=pairs[:-1], test_pairs=[pairs[-1]])


def _gen_rotate_task(task_id: str) -> ARCTask:
    """Generate a 90-degree rotation task."""
    pairs = []
    for _ in range(random.randint(3, 5)):
        s = random.randint(2, 8)
        inp = _random_grid(s, s)
        out = ARCGrid(data=np.rot90(inp.data, 1).astype(np.int32))
        pairs.append(ARCPair(input=inp, output=out))
    return ARCTask(task_id=task_id, train_pairs=pairs[:-1], test_pairs=[pairs[-1]])


def _gen_fill_task(task_id: str) -> ARCTask:
    """Generate a fill-background task (replace 0 with dominant color)."""
    pairs = []
    fill_color = random.randint(1, 9)
    for _ in range(random.randint(3, 5)):
        h, w = random.randint(2, 8), random.randint(2, 8)
        inp = _random_grid(h, w, n_colors=3)
        out_data = inp.data.copy()
        out_data[out_data == 0] = fill_color
        out = ARCGrid(data=out_data)
        pairs.append(ARCPair(input=inp, output=out))
    return ARCTask(task_id=task_id, train_pairs=pairs[:-1], test_pairs=[pairs[-1]])


def _gen_tile_task(task_id: str) -> ARCTask:
    """Generate a 2x2 tiling task."""
    pairs = []
    for _ in range(random.randint(3, 5)):
        h, w = random.randint(2, 5), random.randint(2, 5)
        inp = _random_grid(h, w, n_colors=3)
        out = ARCGrid(data=np.tile(inp.data, (2, 2)).astype(np.int32))
        pairs.append(ARCPair(input=inp, output=out))
    return ARCTask(task_id=task_id, train_pairs=pairs[:-1], test_pairs=[pairs[-1]])


def _gen_color_swap_task(task_id: str) -> ARCTask:
    """Generate a color swap task (swap two specific colors)."""
    c1, c2 = random.sample(range(1, 10), 2)
    pairs = []
    for _ in range(random.randint(3, 5)):
        h, w = random.randint(2, 8), random.randint(2, 8)
        inp = _random_grid(h, w, n_colors=4)
        out_data = inp.data.copy()
        mask1 = out_data == c1
        mask2 = out_data == c2
        out_data[mask1] = c2
        out_data[mask2] = c1
        out = ARCGrid(data=out_data)
        pairs.append(ARCPair(input=inp, output=out))
    return ARCTask(task_id=task_id, train_pairs=pairs[:-1], test_pairs=[pairs[-1]])
