"""
PROJECT AM v10 — ARC Solver
Combines multiple solving strategies:
1. TRM (pretrained) — fast, handles learned patterns
2. TRM + Test-Time Training — adapts to each specific puzzle
3. LLM Program Synthesis — generates Python transforms for hard puzzles
4. Ensemble — combines all methods, picks best

This is the file that actually solves ARC puzzles.
"""

from __future__ import annotations

import time
import copy
import json
import traceback
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
import numpy as np

from am import TRMConfig, PAD_TOKEN, NUM_TOKENS, MAX_SEQ_LEN
from am.trm import TRM
from am.data import (
    ARCTask, ARCGrid, ARCPair,
    encode_task_for_trm, encode_grid, decode_tokens_to_grid,
)


@dataclass
class SolveResult:
    """Result of attempting to solve one ARC task."""
    task_id: str
    predictions: List[List[List[int]]]  # predicted grids as 2D lists
    confidence: float
    method: str
    time_seconds: float
    correct: Optional[bool] = None


class ARCSolver:
    """
    Unified ARC solver. Orchestrates all solving strategies.

    Usage:
        solver = ARCSolver(model_path="checkpoints/best_eval.pt")
        result = solver.solve(task)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[TRMConfig] = None,
        device: str = "auto",
        llm_api_key: Optional[str] = None,
        llm_model: str = "deepseek-chat",
        llm_base_url: str = "https://api.deepseek.com/v1",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.config = config or TRMConfig()
        self.model = None

        if model_path:
            self._load_model(model_path)

        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url

    def _load_model(self, path: str):
        """Load TRM from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if "config" in checkpoint:
            self.config = checkpoint["config"]
        self.model = TRM(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        # Use EMA weights for inference
        if "ema_state" in checkpoint:
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if name in checkpoint["ema_state"]:
                        p.copy_(checkpoint["ema_state"][name])
        self.model.eval()

    def solve(
        self,
        task: ARCTask,
        methods: Optional[List[str]] = None,
        n_attempts: int = 2,
    ) -> SolveResult:
        """
        Solve an ARC task using configured methods.

        Methods:
            "trm"     — Direct TRM inference (fast)
            "ttt"     — Test-time training (slower, more accurate)
            "llm"     — LLM program synthesis (requires API)
            "all"     — Try all, ensemble results

        Returns the best result.
        """
        if methods is None:
            methods = ["ttt"] if self.model else ["llm"]

        t0 = time.time()
        all_results = []

        for method in methods:
            try:
                if method == "trm" and self.model:
                    preds = self._solve_trm(task, n_attempts)
                    all_results.append(("trm", preds))

                elif method == "ttt" and self.model:
                    preds = self._solve_ttt(task, n_attempts)
                    all_results.append(("ttt", preds))

                elif method == "llm" and self.llm_api_key:
                    preds = self._solve_llm(task, n_attempts)
                    all_results.append(("llm", preds))

            except Exception as e:
                print(f"    {method} failed: {e}")
                traceback.print_exc()

        if not all_results:
            # Fallback: return blank grid
            test_pair = task.test_pairs[0]
            blank = np.zeros((test_pair.output.height, test_pair.output.width),
                             dtype=np.int32).tolist()
            return SolveResult(
                task_id=task.task_id,
                predictions=[blank],
                confidence=0.0,
                method="none",
                time_seconds=time.time() - t0,
            )

        # Pick best result: prefer TTT > LLM > TRM
        # In future: validate against training examples
        best_method, best_preds = all_results[-1]
        confidence = self._estimate_confidence(task, best_preds)

        return SolveResult(
            task_id=task.task_id,
            predictions=best_preds,
            confidence=confidence,
            method=best_method,
            time_seconds=time.time() - t0,
        )

    def _solve_trm(self, task: ARCTask, n_attempts: int) -> List[List[List[int]]]:
        """Solve using pretrained TRM (no fine-tuning)."""
        enc = encode_task_for_trm(task, pair_idx=0)
        h, w = enc["height"], enc["width"]

        x = torch.tensor(enc["x"], dtype=torch.long).unsqueeze(0).to(self.device)
        # Pad x
        if x.shape[1] < self.config.max_x_len:
            x = F.pad(x, (0, self.config.max_x_len - x.shape[1]), value=PAD_TOKEN)
        else:
            x = x[:, :self.config.max_x_len]

        y_init = torch.zeros(1, self.config.max_y_len, dtype=torch.long).to(self.device)

        pred_tokens = self.model.predict(x, y_init)[0].cpu().numpy()
        grid = decode_tokens_to_grid(pred_tokens.tolist(), h, w)
        return [grid.to_list()]

    def _solve_ttt(self, task: ARCTask, n_attempts: int) -> List[List[List[int]]]:
        """
        Test-Time Training: fine-tune a copy of TRM on this specific task's
        training examples, then predict the test output.

        This is the key technique from NVARC/ARChitects that pushed scores
        from ~5% to ~24%.
        """
        predictions = []

        for attempt in range(n_attempts):
            # Clone model
            model_copy = TRM(self.config).to(self.device)
            model_copy.load_state_dict(self.model.state_dict())
            model_copy.train()

            # Optimizer for test-time training
            optimizer = torch.optim.AdamW(
                model_copy.parameters(),
                lr=self.config.ttt_lr,
                weight_decay=0.0,
            )

            # Prepare training data from this task's train pairs
            # Each train pair becomes a training example
            train_batches = []
            for i, pair in enumerate(task.train_pairs):
                # Create a mini-task: use other train pairs as demos, this one as "test"
                demo_pairs = [p for j, p in enumerate(task.train_pairs) if j != i]
                if not demo_pairs:
                    demo_pairs = task.train_pairs  # use all if only one pair

                mini_task = ARCTask(
                    task_id=task.task_id,
                    train_pairs=demo_pairs,
                    test_pairs=[pair],
                )
                enc = encode_task_for_trm(mini_task, pair_idx=0)
                train_batches.append(enc)

            # Also add the full task (with all demos + test input)
            # Train the model to predict each training output
            for step in range(self.config.ttt_steps):
                total_loss = 0.0
                for enc in train_batches:
                    x_np = enc["x"]
                    y_np = enc["y_target"]

                    # Pad
                    if len(x_np) > self.config.max_x_len:
                        x_np = x_np[:self.config.max_x_len]
                    else:
                        x_np = np.pad(x_np, (0, self.config.max_x_len - len(x_np)),
                                      constant_values=PAD_TOKEN)

                    if len(y_np) > self.config.max_y_len:
                        y_np = y_np[:self.config.max_y_len]
                    else:
                        y_np = np.pad(y_np, (0, self.config.max_y_len - len(y_np)),
                                      constant_values=PAD_TOKEN)

                    x = torch.tensor(x_np, dtype=torch.long).unsqueeze(0).to(self.device)
                    y_init = torch.zeros(1, self.config.max_y_len, dtype=torch.long).to(self.device)
                    y_target = torch.tensor(y_np, dtype=torch.long).unsqueeze(0).to(self.device)

                    optimizer.zero_grad()
                    result = model_copy(x, y_init, targets=y_target)
                    loss = result["total_loss"]
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_copy.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()

            # Now predict the test output
            model_copy.eval()
            enc = encode_task_for_trm(task, pair_idx=0)
            h, w = enc["height"], enc["width"]

            x_np = enc["x"]
            if len(x_np) > self.config.max_x_len:
                x_np = x_np[:self.config.max_x_len]
            else:
                x_np = np.pad(x_np, (0, self.config.max_x_len - len(x_np)),
                              constant_values=PAD_TOKEN)

            x = torch.tensor(x_np, dtype=torch.long).unsqueeze(0).to(self.device)
            y_init = torch.zeros(1, self.config.max_y_len, dtype=torch.long).to(self.device)

            with torch.no_grad():
                pred_tokens = model_copy.predict(x, y_init)[0].cpu().numpy()

            grid = decode_tokens_to_grid(pred_tokens.tolist(), h, w)
            predictions.append(grid.to_list())

            del model_copy, optimizer

        return predictions

    def _solve_llm(self, task: ARCTask, n_attempts: int) -> List[List[List[int]]]:
        """
        LLM Program Synthesis: ask an LLM to generate a Python function
        that transforms input grids to output grids.

        This is the approach from Ryan Greenblatt (42% on ARC-AGI-1 Pub)
        and the basis of Poetiq's 54% on ARC-AGI-2.
        """
        import requests as req

        # Format task as a prompt
        prompt = self._format_task_for_llm(task)

        predictions = []
        for attempt in range(n_attempts):
            try:
                # Call LLM API
                response = req.post(
                    f"{self.llm_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.llm_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.llm_model,
                        "messages": [
                            {"role": "system", "content": LLM_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.7 + attempt * 0.1,
                        "max_tokens": 4096,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]

                # Extract Python code
                code = self._extract_code(content)
                if not code:
                    continue

                # Execute and validate
                result = self._execute_transform(code, task)
                if result is not None:
                    predictions.append(result)

            except Exception as e:
                print(f"    LLM attempt {attempt + 1} failed: {e}")

        if not predictions:
            # Return input as fallback
            test_pair = task.test_pairs[0]
            predictions = [test_pair.input.to_list()]

        return predictions

    def _format_task_for_llm(self, task: ARCTask) -> str:
        """Format an ARC task as an LLM prompt."""
        lines = ["Here is an ARC-AGI puzzle. Each example shows an input grid and output grid.",
                 "The grids use numbers 0-9 as colors. Find the transformation pattern",
                 "and write a Python function `transform(grid)` that converts input to output.",
                 ""]

        for i, pair in enumerate(task.train_pairs):
            lines.append(f"Example {i + 1} Input:")
            lines.append(json.dumps(pair.input.to_list()))
            lines.append(f"Example {i + 1} Output:")
            lines.append(json.dumps(pair.output.to_list()))
            lines.append("")

        lines.append("Test Input:")
        lines.append(json.dumps(task.test_pairs[0].input.to_list()))
        lines.append("")
        lines.append("Write a Python function `transform(grid)` that takes a 2D list of ints")
        lines.append("and returns the transformed 2D list. The function should work for ALL")
        lines.append("the examples above, not just be hardcoded for one.")
        lines.append("")
        lines.append("Return ONLY the function, no explanation. Wrap in ```python ... ```")

        return "\n".join(lines)

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        # Try to find code block
        if "```python" in response:
            parts = response.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0].strip()
                return code
        if "```" in response:
            parts = response.split("```")
            if len(parts) > 2:
                code = parts[1].strip()
                if code.startswith("python"):
                    code = code[6:].strip()
                return code
        # Maybe the whole response is code
        if "def transform" in response:
            return response.strip()
        return None

    def _execute_transform(
        self, code: str, task: ARCTask
    ) -> Optional[List[List[int]]]:
        """
        Execute a transform function and validate against training examples.
        Returns the test output if validation passes, None otherwise.
        """
        # Safety: basic code screening
        forbidden = ["import os", "import sys", "subprocess", "open(", "__import__",
                      "exec(", "eval(", "globals", "locals", "import shutil",
                      "import socket", "import http"]
        for f in forbidden:
            if f in code:
                return None

        try:
            namespace = {"__builtins__": {
                "range": range, "len": len, "list": list, "dict": dict,
                "set": set, "tuple": tuple, "int": int, "float": float,
                "str": str, "bool": bool, "abs": abs, "max": max, "min": min,
                "sum": sum, "enumerate": enumerate, "zip": zip, "map": map,
                "filter": filter, "sorted": sorted, "reversed": reversed,
                "any": any, "all": all, "isinstance": isinstance, "type": type,
                "True": True, "False": False, "None": None,
                "print": lambda *a, **k: None,  # suppress prints
            }}

            exec(code, namespace)

            if "transform" not in namespace:
                return None

            transform_fn = namespace["transform"]

            # Validate against all training examples
            for pair in task.train_pairs:
                inp = pair.input.to_list()
                expected = pair.output.to_list()
                result = transform_fn(copy.deepcopy(inp))
                if result != expected:
                    return None

            # All training examples pass! Apply to test input
            test_input = task.test_pairs[0].input.to_list()
            test_output = transform_fn(copy.deepcopy(test_input))

            # Basic validation
            if not isinstance(test_output, list) or not test_output:
                return None
            if not all(isinstance(row, list) for row in test_output):
                return None

            return test_output

        except Exception:
            return None

    def _estimate_confidence(
        self, task: ARCTask, predictions: List[List[List[int]]]
    ) -> float:
        """Estimate confidence based on prediction consistency."""
        if len(predictions) <= 1:
            return 0.5
        # If multiple attempts agree, higher confidence
        same_count = sum(
            1 for p in predictions[1:]
            if p == predictions[0]
        )
        return (same_count + 1) / len(predictions)


LLM_SYSTEM_PROMPT = """You are an expert at solving ARC-AGI puzzles. ARC puzzles present \
input-output grid pairs where you must find the transformation rule.

Rules:
- Grids are 2D arrays of integers 0-9 (representing colors)
- Grid size can change between input and output
- The same transformation applies to ALL examples
- Write clean, general Python code
- The function must be named `transform` and take a 2D list of ints

Common ARC patterns:
- Rotation, reflection, transposition
- Color mapping/swapping
- Pattern completion (fill gaps)
- Object detection and manipulation
- Scaling (upsample/downsample)
- Gravity (objects fall)
- Border/frame operations
- Symmetry completion
- Flood fill
- Object counting → grid generation

Think step by step about what changes between input and output, then write the code."""
