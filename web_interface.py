#!/usr/bin/env python3
"""Local web interface for PROJECT AM v10.

This server intentionally uses only the Python standard library so the
dashboard still opens before ML dependencies are fixed.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import platform
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"

DATASETS = {
    "arc1": DATA_ROOT / "arc-agi-1" / "data",
    "arc2": DATA_ROOT / "arc-agi-2" / "data",
}
SPLITS = {"training", "evaluation"}
TASK_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def dependency_status() -> dict[str, dict[str, Any]]:
    deps: dict[str, dict[str, Any]] = {}
    for name in ("numpy", "torch", "requests", "tqdm", "arc_agi"):
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "installed")
            if name == "torch":
                cuda = bool(module.cuda.is_available())
                gpu = module.cuda.get_device_name() if cuda else None
                deps[name] = {"ok": True, "version": version, "cuda": cuda, "gpu": gpu}
            else:
                deps[name] = {"ok": True, "version": version}
        except Exception as exc:
            deps[name] = {"ok": False, "error": str(exc)}
    return deps


def task_dir(dataset: str, split: str) -> Path:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    return DATASETS[dataset] / split


def list_tasks(dataset: str, split: str, limit: int = 200) -> list[dict[str, Any]]:
    directory = task_dir(dataset, split)
    if not directory.exists():
        return []

    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json"))[:limit]:
        try:
            raw = json.loads(path.read_text())
            train_count = len(raw.get("train", []))
            test_count = len(raw.get("test", []))
            first = raw.get("train", [{}])[0].get("input", [])
            shape = f"{len(first)}x{len(first[0])}" if first and first[0] else "?"
        except Exception:
            train_count = 0
            test_count = 0
            shape = "unreadable"
        rows.append(
            {
                "id": path.stem,
                "train": train_count,
                "test": test_count,
                "shape": shape,
                "bytes": path.stat().st_size,
            }
        )
    return rows


def load_task(dataset: str, split: str, task_id: str | None = None) -> dict[str, Any]:
    directory = task_dir(dataset, split)
    if not directory.exists():
        raise FileNotFoundError(f"Missing task directory: {directory}")

    if task_id:
        if not TASK_ID_RE.match(task_id):
            raise ValueError("Invalid task id")
        path = directory / f"{task_id}.json"
    else:
        try:
            path = next(iter(sorted(directory.glob("*.json"))))
        except StopIteration as exc:
            raise FileNotFoundError(f"No tasks in {directory}") from exc

    if not path.exists() or path.parent != directory:
        raise FileNotFoundError(f"Task not found: {task_id}")

    raw = json.loads(path.read_text())
    raw["task_id"] = path.stem
    raw["dataset"] = dataset
    raw["split"] = split
    return raw


def checkpoint_rows() -> list[dict[str, Any]]:
    if not CHECKPOINT_DIR.exists():
        return []
    rows = []
    for path in sorted(CHECKPOINT_DIR.glob("*")):
        if path.is_file():
            rows.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "modified": int(path.stat().st_mtime),
                }
            )
    return rows


def status_payload() -> dict[str, Any]:
    datasets = {}
    for name in DATASETS:
        datasets[name] = {}
        for split in sorted(SPLITS):
            directory = task_dir(name, split)
            datasets[name][split] = {
                "path": str(directory),
                "exists": directory.exists(),
                "tasks": len(list(directory.glob("*.json"))) if directory.exists() else 0,
            }

    return {
        "project": "PROJECT AM v10",
        "root": str(ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "deps": dependency_status(),
        "datasets": datasets,
        "checkpoints": checkpoint_rows(),
        "logs": str(LOG_DIR),
        "recommended_commands": [
            "python3 main.py --test",
            "python3 main.py --quick-test --dataset arc1 -n 5",
            "python3 main.py --evaluate --dataset arc1 --ttt --max-tasks 5",
            "python3 main.py --train --dataset arc1 --epochs 50 --batch-size 4",
        ],
    }


def json_bytes(payload: Any, status: int = 200) -> tuple[int, bytes, str]:
    return status, json.dumps(payload, indent=2).encode("utf-8"), "application/json"


def html_page() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PROJECT AM v10 ARC Console</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #090b10;
      --panel: #111722;
      --panel2: #151f2d;
      --text: #eef4ff;
      --muted: #90a0b8;
      --line: #27344a;
      --ok: #62d18f;
      --warn: #ffcf6a;
      --bad: #ff7b7b;
      --accent: #78a6ff;
      --c0: #050608; --c1: #2e7df6; --c2: #e54949; --c3: #42c774; --c4: #f0d54a;
      --c5: #9298a3; --c6: #d84fd4; --c7: #ff9138; --c8: #43d7d3; --c9: #8b5cf6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 18px 22px;
      border-bottom: 1px solid var(--line);
      background: #0d1119;
      position: sticky;
      top: 0;
      z-index: 10;
    }
    h1 { margin: 0; font-size: 18px; letter-spacing: 0; }
    h2 { margin: 0 0 12px; font-size: 15px; }
    button, select, input {
      border: 1px solid var(--line);
      background: var(--panel2);
      color: var(--text);
      border-radius: 7px;
      padding: 9px 10px;
      font: inherit;
    }
    button { cursor: pointer; }
    button:hover { border-color: var(--accent); }
    main {
      display: grid;
      grid-template-columns: 360px 1fr;
      min-height: calc(100vh - 64px);
    }
    aside {
      border-right: 1px solid var(--line);
      background: #0b1018;
      padding: 16px;
      overflow: auto;
      max-height: calc(100vh - 64px);
    }
    section { padding: 16px; }
    .card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
      margin-bottom: 12px;
    }
    .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .stack { display: grid; gap: 8px; }
    .muted { color: var(--muted); }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid var(--line);
      background: #0c121b;
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 12px;
    }
    .ok { color: var(--ok); }
    .warn { color: var(--warn); }
    .bad { color: var(--bad); }
    .task-list {
      display: grid;
      gap: 6px;
      max-height: 48vh;
      overflow: auto;
    }
    .task-button {
      width: 100%;
      text-align: left;
      display: flex;
      justify-content: space-between;
      gap: 8px;
      background: #0d1420;
    }
    .task-button.active { border-color: var(--accent); background: #13213a; }
    .grid-wrap {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }
    .pair {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #0c121b;
      min-width: 0;
    }
    .pair-title {
      display: flex;
      justify-content: space-between;
      margin-bottom: 8px;
      font-size: 12px;
      color: var(--muted);
    }
    .arc-grid {
      display: grid;
      width: max-content;
      max-width: 100%;
      overflow: hidden;
      border: 1px solid #39455a;
      background: #39455a;
      gap: 1px;
    }
    .cell {
      width: min(22px, 4vw);
      aspect-ratio: 1;
      border-radius: 1px;
    }
    .cmd {
      display: block;
      padding: 10px;
      background: #070a0f;
      border: 1px solid var(--line);
      border-radius: 7px;
      color: #c9dcff;
      overflow: auto;
      white-space: nowrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 320px;
      overflow: auto;
      margin: 0;
      color: #c9dcff;
      font-size: 12px;
    }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
      aside { max-height: none; border-right: 0; border-bottom: 1px solid var(--line); }
      header { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>PROJECT AM v10 ARC Console</h1>
      <div class="muted">Local puzzle browser, system status, and run board</div>
    </div>
    <div class="row">
      <span id="gpu" class="pill">GPU: checking</span>
      <span id="datasets" class="pill">datasets: checking</span>
      <button id="refresh">Refresh</button>
    </div>
  </header>
  <main>
    <aside>
      <div class="card stack">
        <h2>Task Source</h2>
        <div class="row">
          <select id="dataset">
            <option value="arc1">ARC-AGI-1</option>
            <option value="arc2">ARC-AGI-2</option>
          </select>
          <select id="split">
            <option value="training">training</option>
            <option value="evaluation">evaluation</option>
          </select>
        </div>
        <input id="filter" placeholder="Filter task id" />
      </div>
      <div class="card">
        <h2>Tasks</h2>
        <div id="tasks" class="task-list muted">Loading tasks...</div>
      </div>
      <div class="card">
        <h2>Run Commands</h2>
        <div id="commands" class="stack"></div>
      </div>
    </aside>
    <section>
      <div class="card">
        <h2>System</h2>
        <div id="status" class="row muted">Loading status...</div>
      </div>
      <div class="card">
        <h2 id="task-title">Task</h2>
        <div id="task-meta" class="row muted"></div>
      </div>
      <div id="task-view" class="grid-wrap"></div>
      <div class="card">
        <h2>Raw Task JSON</h2>
        <pre id="raw"></pre>
      </div>
    </section>
  </main>
  <script>
    const state = { status: null, tasks: [], active: null };
    const colors = ["var(--c0)", "var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)", "var(--c7)", "var(--c8)", "var(--c9)"];

    const $ = (id) => document.getElementById(id);
    const esc = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

    async function getJSON(url) {
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }

    function depPill(name, dep) {
      if (!dep || !dep.ok) return `<span class="pill bad">${name}: missing</span>`;
      const extra = dep.gpu ? `, ${esc(dep.gpu)}` : "";
      return `<span class="pill ok">${name}: ${esc(dep.version)}${extra}</span>`;
    }

    function renderStatus() {
      const s = state.status;
      const deps = s.deps || {};
      $("gpu").textContent = deps.torch && deps.torch.gpu ? `GPU: ${deps.torch.gpu}` : "GPU: not detected";
      $("gpu").className = deps.torch && deps.torch.gpu ? "pill ok" : "pill warn";
      const total = Object.values(s.datasets).flatMap(v => Object.values(v)).reduce((n, d) => n + d.tasks, 0);
      $("datasets").textContent = `datasets: ${total} tasks`;
      $("datasets").className = total ? "pill ok" : "pill bad";
      $("status").innerHTML = [
        depPill("torch", deps.torch),
        depPill("numpy", deps.numpy),
        depPill("arc_agi", deps.arc_agi),
        `<span class="pill">checkpoints: ${s.checkpoints.length}</span>`,
        `<span class="pill">root: ${esc(s.root)}</span>`
      ].join("");
      $("commands").innerHTML = s.recommended_commands.map(cmd => `<code class="cmd">${esc(cmd)}</code>`).join("");
    }

    function gridHTML(grid) {
      if (!Array.isArray(grid) || !grid.length) return "<div class='muted'>empty grid</div>";
      const cols = Math.max(...grid.map(row => row.length));
      const cells = grid.flatMap(row => row.map(v => `<span class="cell" title="${v}" style="background:${colors[v] || "#fff"}"></span>`)).join("");
      return `<div class="arc-grid" style="grid-template-columns: repeat(${cols}, 1fr)">${cells}</div>`;
    }

    function pairHTML(kind, idx, pair) {
      const out = pair.output ? gridHTML(pair.output) : "<div class='muted'>hidden / unavailable</div>";
      return `<div class="pair">
        <div class="pair-title"><strong>${kind} ${idx + 1}</strong><span>input</span></div>
        ${gridHTML(pair.input)}
        <div class="pair-title" style="margin-top:10px"><span></span><span>output</span></div>
        ${out}
      </div>`;
    }

    function renderTask(task) {
      state.active = task.task_id;
      $("task-title").textContent = `${task.dataset.toUpperCase()} / ${task.split} / ${task.task_id}`;
      $("task-meta").innerHTML = [
        `<span class="pill">train pairs: ${task.train.length}</span>`,
        `<span class="pill">test pairs: ${task.test.length}</span>`
      ].join("");
      $("task-view").innerHTML = [
        ...task.train.map((p, i) => pairHTML("train", i, p)),
        ...task.test.map((p, i) => pairHTML("test", i, p))
      ].join("");
      $("raw").textContent = JSON.stringify(task, null, 2);
      renderTasks();
    }

    function renderTasks() {
      const q = $("filter").value.trim().toLowerCase();
      const rows = state.tasks.filter(t => !q || t.id.toLowerCase().includes(q));
      $("tasks").innerHTML = rows.map(t => {
        const active = t.id === state.active ? " active" : "";
        return `<button class="task-button${active}" data-id="${esc(t.id)}">
          <span>${esc(t.id)}</span>
          <span class="muted">${t.train}/${t.test} ${esc(t.shape)}</span>
        </button>`;
      }).join("") || "<div class='muted'>No tasks found</div>";
      document.querySelectorAll(".task-button").forEach(btn => {
        btn.onclick = () => loadTask(btn.dataset.id);
      });
    }

    async function loadTasks() {
      const dataset = $("dataset").value;
      const split = $("split").value;
      state.tasks = await getJSON(`/api/tasks?dataset=${dataset}&split=${split}&limit=300`);
      renderTasks();
      if (state.tasks[0]) await loadTask(state.tasks[0].id);
    }

    async function loadTask(id) {
      const dataset = $("dataset").value;
      const split = $("split").value;
      const task = await getJSON(`/api/task?dataset=${dataset}&split=${split}&id=${encodeURIComponent(id)}`);
      renderTask(task);
    }

    async function refresh() {
      state.status = await getJSON("/api/status");
      renderStatus();
      await loadTasks();
    }

    $("dataset").onchange = loadTasks;
    $("split").onchange = loadTasks;
    $("filter").oninput = renderTasks;
    $("refresh").onclick = refresh;
    refresh().catch(err => {
      $("status").innerHTML = `<span class="bad">${esc(err.message)}</span>`;
    });
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "AMArcConsole/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def send_payload(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                body = html_page().encode("utf-8")
                self.send_payload(200, body, "text/html; charset=utf-8")
            elif parsed.path == "/api/status":
                self.send_payload(*json_bytes(status_payload()))
            elif parsed.path == "/api/health":
                status = status_payload()
                payload = {
                    "ok": True,
                    "project": status["project"],
                    "datasets": status["datasets"],
                    "torch": status["deps"].get("torch", {}),
                    "checkpoints": len(status["checkpoints"]),
                }
                self.send_payload(*json_bytes(payload))
            elif parsed.path == "/api/tasks":
                dataset = params.get("dataset", ["arc1"])[0]
                split = params.get("split", ["training"])[0]
                limit = int(params.get("limit", ["200"])[0])
                self.send_payload(*json_bytes(list_tasks(dataset, split, limit)))
            elif parsed.path == "/api/task":
                dataset = params.get("dataset", ["arc1"])[0]
                split = params.get("split", ["training"])[0]
                task_id = params.get("id", [None])[0]
                self.send_payload(*json_bytes(load_task(dataset, split, task_id)))
            else:
                body = json.dumps({"error": "not found", "path": html.escape(parsed.path)}).encode("utf-8")
                self.send_payload(404, body, "application/json")
        except Exception as exc:
            body = json.dumps({"error": str(exc)}, indent=2).encode("utf-8")
            self.send_payload(400, body, "application/json")


def main() -> None:
    parser = argparse.ArgumentParser(description="PROJECT AM v10 local ARC web console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AM_ARC_PORT", "7860")))
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"PROJECT AM v10 ARC Console: {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping ARC Console.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
