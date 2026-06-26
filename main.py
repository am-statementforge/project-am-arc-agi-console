#!/usr/bin/env python3
"""
PROJECT AM v10 — Main Entry Point
══════════════════════════════════

Usage:
    # First run: download data and test everything works
    python main.py --test

    # Download ARC datasets
    python main.py --download

    # Train TRM on ARC-AGI-1
    python main.py --train --dataset arc1

    # Train TRM on ARC-AGI-2
    python main.py --train --dataset arc2

    # Evaluate with test-time training
    python main.py --evaluate --dataset arc1 --ttt

    # Evaluate with LLM program synthesis (needs API key)
    python main.py --evaluate --dataset arc2 --llm --api-key YOUR_KEY

    # Quick test on 5 tasks
    python main.py --quick-test
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def cmd_test(args):
    """Test that everything works."""
    print("\n" + "=" * 60)
    print("  PROJECT AM v10 — System Test")
    print("=" * 60)

    errors = []

    # 1. Python version
    print(f"\n  Python: {sys.version}")

    # 2. PyTorch
    try:
        import torch
        gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "None (CPU only)"
        print(f"  PyTorch: {torch.__version__}")
        print(f"  GPU: {gpu}")
    except ImportError:
        errors.append("PyTorch not installed. Run: pip install torch")
        print("  PyTorch: NOT INSTALLED")

    # 3. NumPy
    try:
        import numpy as np
        print(f"  NumPy: {np.__version__}")
    except ImportError:
        errors.append("NumPy not installed. Run: pip install numpy")

    # 4. TRM model
    try:
        from am import TRMConfig
        from am.trm import TRM
        config = TRMConfig(dim=64, num_layers=2, num_heads=4, H_cycles=2, L_cycles=3,
                           max_x_len=256, max_y_len=128)
        model = TRM(config)
        n_params = sum(p.numel() for p in model.parameters())

        # Test forward pass
        import torch
        x = torch.randint(0, 10, (1, 256))
        y = torch.zeros(1, 128, dtype=torch.long)
        result = model(x, y, targets=y)
        print(f"  TRM forward pass: OK (loss={result['total_loss'].item():.4f})")
    except Exception as e:
        errors.append(f"TRM failed: {e}")
        print(f"  TRM: FAILED ({e})")

    # 5. Data pipeline
    try:
        from am.data import generate_synthetic_tasks, encode_task_for_trm
        tasks = generate_synthetic_tasks(5)
        enc = encode_task_for_trm(tasks[0])
        print(f"  Data pipeline: OK ({len(tasks)} synthetic tasks, encoded x={len(enc['x'])} tokens)")
    except Exception as e:
        errors.append(f"Data pipeline failed: {e}")
        print(f"  Data pipeline: FAILED ({e})")

    # 6. Check ARC data
    from am import DATA_DIR
    for v in ["arc-agi-1", "arc-agi-2"]:
        p = DATA_DIR / v
        if p.exists():
            n = len(list(p.rglob("*.json")))
            print(f"  {v}: {n} files")
        else:
            print(f"  {v}: Not downloaded (run: python main.py --download)")

    # 7. ARC-AGI-3 toolkit
    try:
        import arc_agi
        print(f"  ARC-AGI-3 toolkit: OK")
    except ImportError:
        print(f"  ARC-AGI-3 toolkit: Not installed (run: pip install arc-agi)")

    # Summary
    print(f"\n{'='*60}")
    if errors:
        print(f"  {len(errors)} ERRORS found:")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  ALL SYSTEMS GO")
    print(f"{'='*60}\n")


def cmd_download(args):
    """Download ARC datasets."""
    from am.data import download_arc_data

    print("\n  Downloading ARC datasets...\n")
    download_arc_data("arc1", force=args.force)
    download_arc_data("arc2", force=args.force)
    print("\n  Done!")


def cmd_train(args):
    """Train TRM on ARC data."""
    from am import TRMConfig
    from am.train import train

    config = TRMConfig(
        dim=args.dim,
        num_layers=args.layers,
        num_heads=args.heads,
        H_cycles=args.h_cycles,
        L_cycles=args.l_cycles,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    train(
        config=config,
        dataset_version=args.dataset,
        device=args.device,
        resume_from=args.resume,
        synthetic=args.synthetic,
    )


def cmd_evaluate(args):
    """Evaluate solver on ARC dataset."""
    from am import TRMConfig, CHECKPOINT_DIR
    from am.solver import ARCSolver
    from am.evaluate import evaluate

    # Find best checkpoint
    model_path = args.model
    if not model_path:
        for candidate in ["best_eval.pt", "final.pt"]:
            p = CHECKPOINT_DIR / candidate
            if p.exists():
                model_path = str(p)
                break

    methods = []
    if args.ttt:
        methods.append("ttt")
    elif model_path:
        methods.append("trm")
    if args.llm:
        methods.append("llm")
    if not methods:
        methods = ["ttt"] if model_path else ["llm"]

    solver = ARCSolver(
        model_path=model_path,
        device=args.device,
        llm_api_key=args.api_key or os.environ.get("DEEPSEEK_API_KEY"),
        llm_model=args.llm_model,
    )

    evaluate(
        solver=solver,
        dataset_version=args.dataset,
        split=args.split,
        methods=methods,
        n_attempts=args.attempts,
        max_tasks=args.max_tasks,
    )


def cmd_quick_test(args):
    """Quick test on a few training tasks."""
    from am import TRMConfig, CHECKPOINT_DIR
    from am.solver import ARCSolver
    from am.evaluate import quick_test

    model_path = args.model
    if not model_path:
        for candidate in ["best_eval.pt", "final.pt"]:
            p = CHECKPOINT_DIR / candidate
            if p.exists():
                model_path = str(p)
                break

    if not model_path:
        print("\n  No trained model found. Training a small model first...\n")
        from am.train import train
        config = TRMConfig(dim=128, epochs=200, batch_size=2,
                           max_x_len=512, max_y_len=256)
        train(config=config, dataset_version="arc1", synthetic=50)
        model_path = str(CHECKPOINT_DIR / "final.pt")

    solver = ARCSolver(model_path=model_path, device=args.device)
    quick_test(solver, dataset_version=args.dataset, n=args.n)


def main():
    parser = argparse.ArgumentParser(
        description="PROJECT AM v10 — ARC-AGI Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # Also support flat flags for simplicity
    parser.add_argument("--test", action="store_true", help="Run system test")
    parser.add_argument("--download", action="store_true", help="Download ARC datasets")
    parser.add_argument("--train", action="store_true", help="Train TRM")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate solver")
    parser.add_argument("--quick-test", action="store_true", help="Quick test on 5 tasks")

    # Common args
    parser.add_argument("--dataset", default="arc1", choices=["arc1", "arc2"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", default=None, help="Path to model checkpoint")

    # Training args
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--h-cycles", type=int, default=3)
    parser.add_argument("--l-cycles", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--synthetic", type=int, default=200)
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")

    # Eval args
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--ttt", action="store_true", help="Use test-time training")
    parser.add_argument("--llm", action="store_true", help="Use LLM program synthesis")
    parser.add_argument("--api-key", default=None, help="LLM API key")
    parser.add_argument("--llm-model", default="deepseek-chat")
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("-n", type=int, default=5, help="Number of tasks for quick-test")

    # Download args
    parser.add_argument("--force", action="store_true", help="Force re-download")

    args = parser.parse_args()

    if args.test:
        cmd_test(args)
    elif args.download:
        cmd_download(args)
    elif args.train:
        cmd_train(args)
    elif args.evaluate:
        cmd_evaluate(args)
    elif args.quick_test:
        cmd_quick_test(args)
    else:
        parser.print_help()
        print("\n  Start with: python main.py --test")


if __name__ == "__main__":
    main()
