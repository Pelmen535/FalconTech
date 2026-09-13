import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from vehicle_reid.cli import main
from vehicle_reid.data import Record, audit_records, image_to_array, load_manifest, load_vehicle_image
from vehicle_reid.demo_data import make_demo
from vehicle_reid.features import ColorGridEmbedder
from vehicle_reid.retrieval import evaluate


class PipelineTests(unittest.TestCase):
    def test_demo_pipeline_retrieves_cross_camera_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = make_demo(temp)
            records = load_manifest(manifest)
            report = audit_records(records)
            self.assertEqual(report.queries, 3)
            self.assertEqual(report.positive_queries, 3)
            queries = [row for row in records if row.split == "query"]
            gallery = [row for row in records if row.split == "gallery"]
            embedder = ColorGridEmbedder()
            metrics, _ = evaluate(queries, embedder.embed(queries), gallery, embedder.embed(gallery))
            self.assertEqual(metrics["recall_at_1"], 1.0)

    def test_visible_plate_is_masked_before_crop(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "frame.png"
            Image.new("RGB", (20, 20), (255, 255, 255)).save(image_path)
            record = Record(image_path, "v1", "c1", "query", "visible", (4, 5, 12, 10), None)
            array = image_to_array(load_vehicle_image(record))
            expected = np.asarray([127, 127, 127], dtype=np.float32) / 255.0
            self.assertTrue(np.allclose(array[7, 8], expected))

    def test_train_eval_identity_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root / "train.png"
            query = root / "query.png"
            Image.new("RGB", (4, 4), "red").save(train)
            Image.new("RGB", (4, 4), "blue").save(query)
            records = [
                Record(train, "same", "c1", "train", "absent"),
                Record(query, "same", "c2", "query", "absent"),
            ]
            with self.assertRaisesRegex(ValueError, "identity leakage"):
                audit_records(records)

    def test_visible_plate_without_bbox_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "image.png"
            Image.new("RGB", (4, 4), "red").save(image)
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["image_path", "vehicle_id", "camera_id", "split", "plate_status", "plate_bbox"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_path": image.name,
                        "vehicle_id": "v1",
                        "camera_id": "c1",
                        "split": "query",
                        "plate_status": "visible",
                        "plate_bbox": "",
                    }
                )
            with self.assertRaisesRegex(ValueError, "visible plate requires plate_bbox"):
                load_manifest(manifest)

    def test_cli_writes_metrics_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = make_demo(root / "data")
            output = root / "metrics.json"
            self.assertEqual(main(["evaluate", "--manifest", str(manifest), "--out", str(output)]), 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["metrics"]["evaluated_queries"], 3)


if __name__ == "__main__":
    unittest.main()
