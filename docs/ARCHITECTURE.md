# Architecture

PROJECT AM v10 is a local-first ARC-AGI research system.

## Layers

1. Frontend

   The browser UI is embedded in `web_interface.py` and served as a single-page app. It visualizes ARC grids, task metadata, dependency status, GPU status, checkpoints, and run commands.

2. Backend

   `web_interface.py` exposes JSON endpoints for health, status, task listing, and task loading. It uses only the standard library so the dashboard can boot even before ML dependencies are installed.

3. Data Pipeline

   `am/data.py` downloads ARC-AGI-1 and ARC-AGI-2, loads tasks, encodes grids into token sequences, and generates synthetic/augmented training tasks.

4. Model

   `am/trm.py` implements the Tiny Recursive Model style solver: embeddings, recursive latent updates, adaptive computation, and output decoding.

5. Training and Evaluation

   `am/train.py` handles training, EMA, checkpointing, and logs. `am/evaluate.py` evaluates tasks and writes detailed result JSON.

6. Solver Orchestration

   `am/solver.py` combines direct TRM inference, test-time training, and optional LLM synthesis behind a single `ARCSolver` interface.

## Local-First Safety

The repo intentionally excludes:

- downloaded datasets
- model checkpoints
- logs
- local virtualenvs
- run artifacts

This keeps GitHub clean while preserving reproducible commands for rebuilding local state.
