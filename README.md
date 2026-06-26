# PROJECT AM v10 - ARC-AGI Research Console

Full-stack local research console for ARC-AGI experiments. It combines a Python backend, a zero-dependency browser UI, a PyTorch TRM solver, dataset tooling, checkpoint awareness, CI, Docker support, and reproducible tests.

This repo is designed to show practical AI engineering: dataset handling, model orchestration, API design, frontend visualization, safe local defaults, and developer-grade packaging.

## Quick Start

```bash
git clone https://github.com/am-statementforge/project-am-arc-agi-console.git
cd project-am-arc-agi-console
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py --download
python3 main.py --test
python3 web_interface.py --port 7860
```

Open the console:

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

The ARC datasets and checkpoints are intentionally not committed. `python3 main.py --download` restores the public ARC-AGI task files locally.

## Backend API

```text
GET /api/health
GET /api/status
GET /api/tasks?dataset=arc1&split=training&limit=200
GET /api/task?dataset=arc1&split=training&id=007bbfb7
```

## Solver Commands

```bash
python3 main.py --test
python3 main.py --quick-test --dataset arc1 -n 5
python3 main.py --evaluate --dataset arc1 --ttt --max-tasks 5
python3 main.py --train --dataset arc1 --epochs 50 --batch-size 4
```

## Engineering Quality

```bash
make test
make serve
```

Included:

- `tests/` for backend and dataset API behavior
- `.github/workflows/ci.yml` for syntax and unit checks
- `Dockerfile` for reproducible local serving
- `docs/API.md` and `docs/ARCHITECTURE.md` for system design review

## Publish Notes

Before uploading, keep generated and heavy local assets out of git:

- `data/`
- `checkpoints/`
- `logs/`
- `venv/` or `.venv/`
- `__pycache__/`

Dataset download links live in `am/data.py`; the repo should publish code and docs, not local downloaded datasets or checkpoints.
