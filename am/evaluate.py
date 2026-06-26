"""
PROJECT AM v10 — Evaluation
Evaluates the solver on ARC-AGI-1/2 datasets and reports scores.
This is the file that produces the number we care about.
"""

from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable

from am import CHECKPOINT_DIR, LOG_DIR
from am.data import load_arc_dataset, ARCTask
from am.solver import ARCSolver, SolveResult


def evaluate(
    solver: ARCSolver,
    dataset_version: str = "arc1",
    split: str = "evaluation",
    methods: Optional[List[str]] = None,
    n_attempts: int = 2,
    max_tasks: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate solver on ARC dataset. Returns detailed results.

    The score is: (number of tasks where at least one prediction
    exactly matches the expected output) / (total tasks).
    """
    print(f"\n{'='*60}")
    print(f"  EVALUATING on {dataset_version.upper()} ({split})")
    print(f"{'='*60}\n")

    tasks = load_arc_dataset(dataset_version, split=split)
    if max_tasks:
        tasks = tasks[:max_tasks]

    print(f"  Tasks: {len(tasks)}")
    print(f"  Methods: {methods or 'default'}")
    print(f"  Attempts per task: {n_attempts}\n")

    results = []
    correct = 0
    total = 0
    method_stats = {}

    iterator = tqdm(tasks, desc="  Solving") if verbose else tasks
    for task in iterator:
        result = solver.solve(task, methods=methods, n_attempts=n_attempts)

        # Check if any prediction matches expected output
        is_correct = False
        for test_pair in task.test_pairs:
            expected = test_pair.output.to_list()
            for pred in result.predictions:
                if pred == expected:
                    is_correct = True
                    break
            if is_correct:
                break

        result.correct = is_correct
        results.append(result)

        if is_correct:
            correct += 1
        total += 1

        # Track per-method stats
        m = result.method
        if m not in method_stats:
            method_stats[m] = {"correct": 0, "total": 0, "time": 0.0}
        method_stats[m]["total"] += 1
        method_stats[m]["time"] += result.time_seconds
        if is_correct:
            method_stats[m]["correct"] += 1

    accuracy = correct / max(total, 1)

    # Report
    print(f"\n{'='*60}")
    print(f"  RESULTS: {dataset_version.upper()} ({split})")
    print(f"{'='*60}")
    print(f"\n  Score: {correct}/{total} = {accuracy:.1%}\n")

    for m, stats in method_stats.items():
        m_acc = stats["correct"] / max(stats["total"], 1)
        m_avg_time = stats["time"] / max(stats["total"], 1)
        print(f"  Method '{m}': {stats['correct']}/{stats['total']} = {m_acc:.1%}, "
              f"avg {m_avg_time:.2f}s/task")

    # Save detailed results
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results_path = LOG_DIR / f"eval_{dataset_version}_{split}_{int(time.time())}.json"
    serializable = []
    for r in results:
        serializable.append({
            "task_id": r.task_id,
            "correct": r.correct,
            "method": r.method,
            "confidence": r.confidence,
            "time": round(r.time_seconds, 3),
            "n_predictions": len(r.predictions),
        })
    with open(results_path, "w") as f:
        json.dump({
            "dataset": dataset_version,
            "split": split,
            "score": accuracy,
            "correct": correct,
            "total": total,
            "method_stats": method_stats,
            "tasks": serializable,
        }, f, indent=2)

    print(f"\n  Detailed results saved to {results_path}")

    return {
        "score": accuracy,
        "correct": correct,
        "total": total,
        "method_stats": method_stats,
        "results": results,
    }


def quick_test(solver: ARCSolver, dataset_version: str = "arc1", n: int = 5):
    """Quick test on a few tasks to verify everything works."""
    print(f"\n  Quick test: {n} tasks from {dataset_version} training set\n")

    tasks = load_arc_dataset(dataset_version, split="training")[:n]

    for task in tasks:
        result = solver.solve(task, methods=["ttt"], n_attempts=1)

        expected = task.test_pairs[0].output.to_list()
        is_correct = any(p == expected for p in result.predictions)

        status = "CORRECT" if is_correct else "WRONG"
        print(f"  {task.task_id}: {status} ({result.method}, {result.time_seconds:.1f}s, "
              f"conf={result.confidence:.2f})")

        if not is_correct and result.predictions:
            pred = np.array(result.predictions[0])
            exp = np.array(expected)
            if pred.shape == exp.shape:
                match_pct = (pred == exp).mean()
                print(f"    Pixel match: {match_pct:.1%}")
            else:
                print(f"    Shape mismatch: predicted {pred.shape} vs expected {exp.shape}")
