# PROJECT AM v10 - ARC-AGI Solver

Local ARC-AGI research repo with a PyTorch TRM solver, ARC-AGI-1/2 datasets, checkpoint support, and a zero-dependency web console.

## Quick Start

```bash
cd project-am-v10
python3 main.py --test
python3 web_interface.py --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

## Interface

The local interface shows:

- dependency and GPU status
- ARC-AGI-1 and ARC-AGI-2 task counts
- checkpoint inventory
- visual task browser for training/evaluation splits
- copyable commands for test, quick-test, evaluation, and GPU training

It uses only the Python standard library, so it opens even when ML packages or a virtualenv are broken.

## Solver Commands

```bash
python3 main.py --test
python3 main.py --quick-test --dataset arc1 -n 5
python3 main.py --evaluate --dataset arc1 --ttt --max-tasks 5
python3 train_gpu.py --dataset arc1
```

## Publish Notes

Before uploading, keep generated and heavy local assets out of git:

- `data/`
- `checkpoints/`
- `logs/`
- `venv/` or `.venv/`
- `__pycache__/`

Dataset download links live in `am/data.py`; the repo should publish code and docs, not local downloaded datasets or checkpoints.
