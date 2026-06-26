import json
import tempfile
import unittest
from pathlib import Path

import web_interface


class WebInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.data = root / "data"
        self.checkpoints = root / "checkpoints"
        self.logs = root / "logs"
        task_dir = self.data / "arc-agi-1" / "data" / "training"
        task_dir.mkdir(parents=True)
        self.checkpoints.mkdir()
        self.logs.mkdir()
        (task_dir / "demo123.json").write_text(
            json.dumps(
                {
                    "train": [
                        {
                            "input": [[1, 0], [0, 1]],
                            "output": [[1, 1], [1, 1]],
                        }
                    ],
                    "test": [
                        {
                            "input": [[0, 1], [1, 0]],
                            "output": [[1, 1], [1, 1]],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        self.old_datasets = web_interface.DATASETS
        self.old_checkpoints = web_interface.CHECKPOINT_DIR
        self.old_logs = web_interface.LOG_DIR
        web_interface.DATASETS = {
            "arc1": self.data / "arc-agi-1" / "data",
            "arc2": self.data / "arc-agi-2" / "data",
        }
        web_interface.CHECKPOINT_DIR = self.checkpoints
        web_interface.LOG_DIR = self.logs

    def tearDown(self):
        web_interface.DATASETS = self.old_datasets
        web_interface.CHECKPOINT_DIR = self.old_checkpoints
        web_interface.LOG_DIR = self.old_logs
        self.tmp.cleanup()

    def test_list_tasks_reads_arc_metadata(self):
        rows = web_interface.list_tasks("arc1", "training")
        self.assertEqual(rows[0]["id"], "demo123")
        self.assertEqual(rows[0]["train"], 1)
        self.assertEqual(rows[0]["test"], 1)
        self.assertEqual(rows[0]["shape"], "2x2")

    def test_load_task_includes_identity_fields(self):
        task = web_interface.load_task("arc1", "training", "demo123")
        self.assertEqual(task["task_id"], "demo123")
        self.assertEqual(task["dataset"], "arc1")
        self.assertEqual(task["split"], "training")

    def test_status_payload_is_json_ready(self):
        payload = web_interface.status_payload()
        json.dumps(payload)
        self.assertEqual(payload["datasets"]["arc1"]["training"]["tasks"], 1)

    def test_bad_task_id_is_rejected(self):
        with self.assertRaises(ValueError):
            web_interface.load_task("arc1", "training", "../bad")


if __name__ == "__main__":
    unittest.main()
