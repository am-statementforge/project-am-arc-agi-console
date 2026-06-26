# API

The web backend is served by `web_interface.py` using Python's standard library.

## `GET /api/health`

Small health payload for uptime checks.

Returns:

```json
{
  "ok": true,
  "project": "PROJECT AM v10",
  "datasets": {},
  "torch": {},
  "checkpoints": 0
}
```

## `GET /api/status`

Full project status:

- Python and platform details
- dependency versions
- CUDA/GPU detection when PyTorch is installed
- ARC-AGI dataset counts
- checkpoint inventory
- recommended local commands

## `GET /api/tasks`

Query parameters:

- `dataset`: `arc1` or `arc2`
- `split`: `training` or `evaluation`
- `limit`: number of task rows to return

Example:

```text
/api/tasks?dataset=arc1&split=training&limit=10
```

## `GET /api/task`

Query parameters:

- `dataset`: `arc1` or `arc2`
- `split`: `training` or `evaluation`
- `id`: task id without `.json`

Example:

```text
/api/task?dataset=arc1&split=training&id=007bbfb7
```
