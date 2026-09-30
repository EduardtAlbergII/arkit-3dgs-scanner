#!/usr/bin/env python3
"""COLMAP export contract tests; COLMAP_INTEGRATION=1 also runs a real CPU reconstruction."""

import functools
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import colmap_sfm


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.scan = self.root / "scan with spaces"
        (self.scan / "images").mkdir(parents=True)
        self.output = self.root / "new dataset"
        self.records = [
            {"id": index + 1, "imageFile": f"frame {index}.jpg",
             "intrinsics": {"width": 640, "height": 480, "fx": 500 + index,
                            "fy": 502 + index, "cx": 319, "cy": 239},
             # Deliberately invalid poses: COLMAP must never use them.
             "transform": ["not a pose"]}
            for index in range(5)
        ]
        for record in self.records:
            (self.scan / "images" / record["imageFile"]).write_bytes(b"original photo")
        (self.scan / "capture-meta.json").write_bytes(b'{"unknown":"preserve me"}\n')
        (self.scan / "points.ply").write_bytes(b"ARKit frame; never mix into COLMAP")
        self.write_records()
        self.calls = []
        self.model_names = {"0": [record["imageFile"] for record in self.records]}

    def write_records(self):
        (self.scan / "poses_refined.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in self.records))

    def create_database(self, database):
        with sqlite3.connect(database) as connection:
            connection.executescript(
                "CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY, model INTEGER, width INTEGER, "
                "height INTEGER, params BLOB, prior_focal_length INTEGER);"
                "CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER);"
            )
            for index, record in enumerate(self.records, 1):
                k = record["intrinsics"]
                connection.execute("INSERT INTO cameras VALUES(?, 1, ?, ?, ?, 0)",
                                   (index, k["width"], k["height"], b"uncalibrated"))
                connection.execute("INSERT INTO images VALUES(?, ?, ?)",
                                   (index, record["imageFile"], index))

    def fake_run(self, command, *arguments):
        self.calls.append((command, arguments))
        args = dict(zip(arguments[::2], arguments[1::2]))
        if command == "feature_extractor":
            self.create_database(args["--database_path"])
            self.assertEqual(args["--ImageReader.single_camera_per_image"], 1)
            self.assertEqual(sorted(p.name for p in args["--image_path"].iterdir()),
                             sorted(record["imageFile"] for record in self.records))
        elif command.endswith("_matcher"):
            with sqlite3.connect(args["--database_path"]) as connection:
                cameras = connection.execute("SELECT params, prior_focal_length FROM cameras ORDER BY camera_id").fetchall()
            for record, (params, prior) in zip(self.records, cameras):
                self.assertEqual(struct.unpack("<4d", params),
                                 tuple(record["intrinsics"][key] for key in ("fx", "fy", "cx", "cy")))
                self.assertEqual(prior, 1)
        elif command == "mapper":
            self.assertNotIn("--input_path", args)
            for name in self.model_names:
                model = args["--output_path"] / name
                model.mkdir()
                for filename in ("cameras.bin", "images.bin", "points3D.bin"):
                    (model / filename).write_bytes(b"COLMAP binary fixture")
            self.assertEqual(args["--Mapper.ba_refine_focal_length"], 0)
        elif command == "model_converter":
            model = args["--output_path"]
            names = self.model_names[model.name]
            (model / "images.txt").write_text(
                "# Image list\n" + "".join(f"{i} 1 0 0 0 0 0 0 {i} {name}\n\n"
                                          for i, name in enumerate(names, 1)))
            (model / "points3D.txt").write_text("# Point list\n1 0 0 3 100 120 140 0.5 1 0 2 0\n")
        elif command == "image_undistorter":
            dataset = args["--output_path"]
            (dataset / "images").mkdir(parents=True)
            (dataset / "sparse").mkdir()
            for name in self.model_names[args["--input_path"].name]:
                shutil.copyfile(args["--image_path"] / name, dataset / "images" / name)
            for filename in ("cameras.bin", "images.bin", "points3D.bin"):
                shutil.copyfile(args["--input_path"] / filename, dataset / "sparse" / filename)
            for filename in ("rigs.bin", "frames.bin"):
                (dataset / "sparse" / filename).write_bytes(b"rig/frame fixture")

    def run_fake(self, **kwargs):
        with patch("colmap_sfm.shutil.which", return_value="/usr/bin/colmap"), \
                patch.object(colmap_sfm.Colmap, "option", side_effect=lambda command, *names: "--" + names[-1]), \
                patch.object(colmap_sfm.Colmap, "run", side_effect=self.fake_run):
            return colmap_sfm.reconstruct(self.scan, self.output, **kwargs)

    def snapshot(self):
        return {str(p.relative_to(self.scan)): p.read_bytes() for p in self.scan.rglob("*") if p.is_file()}

    def test_reconstruction_preserves_input_and_uses_independent_geometry(self):
        before = self.snapshot()
        (self.scan / "images" / "not selected.jpg").write_bytes(b"not selected")
        before["images/not selected.jpg"] = b"not selected"
        report = self.run_fake()
        self.assertEqual(before, self.snapshot())
        self.assertEqual(report["inputImages"], 5)
        self.assertEqual(report["meanReprojectionErrorPx"], 0.5)
        self.assertFalse(report["metricScale"])
        self.assertEqual(report["unregisteredImages"], [])
        self.assertEqual(report, json.loads((self.output / "sfm-report.json").read_text()))
        self.assertFalse((self.output / "points.ply").exists())
        self.assertFalse((self.output / "poses_refined.jsonl").exists())
        self.assertTrue((self.output / "sparse/0/images.bin").is_file())
        self.assertTrue((self.output / "sparse/0/rigs.bin").is_file())
        self.assertTrue((self.output / "sparse/0/frames.bin").is_file())

    def test_largest_connected_model_and_missing_images_are_reported(self):
        names = [record["imageFile"] for record in self.records]
        self.model_names = {"0": names[:3], "1": names[1:]}
        report = self.run_fake()
        self.assertEqual(report["registeredImages"], names[1:])
        self.assertEqual(report["unregisteredImages"], names[:1])
        self.assertEqual(report["connectedModels"], 2)
        self.assertFalse((self.output / "images" / names[0]).exists())

    def test_insufficient_registration_is_not_published(self):
        self.model_names["0"] = self.model_names["0"][:3]
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "registered 3/5"):
            self.run_fake()
        self.assertFalse(self.output.exists())
        self.assertEqual(before, self.snapshot())
        self.assertEqual([], list(self.root.glob(".colmap-sfm-*")))

    def test_explicit_partial_registration_opt_in(self):
        self.model_names["0"] = self.model_names["0"][:3]
        self.assertEqual(len(self.run_fake(min_registered_fraction=0.6)["registeredImages"]), 3)

    def test_no_model_fails_without_falling_back_to_arkit(self):
        self.model_names = {}
        with self.assertRaisesRegex(ValueError, "could not reconstruct"):
            self.run_fake()
        self.assertFalse(self.output.exists())

    def test_external_failure_preserves_input_and_cleans_staging(self):
        before = self.snapshot()
        with patch("colmap_sfm.shutil.which", return_value="/usr/bin/colmap"), \
                patch.object(colmap_sfm.Colmap, "option", return_value="--SiftExtraction.use_gpu"), \
                patch.object(colmap_sfm.Colmap, "run", side_effect=subprocess.CalledProcessError(1, "colmap")):
            with self.assertRaises(subprocess.CalledProcessError):
                colmap_sfm.reconstruct(self.scan, self.output)
        self.assertEqual(before, self.snapshot())
        self.assertFalse(self.output.exists())
        self.assertEqual([], list(self.root.glob(".colmap-sfm-*")))

    def test_sequential_matching_is_explicit(self):
        report = self.run_fake(matching="sequential")
        command, args = next(call for call in self.calls if call[0].endswith("_matcher"))
        self.assertEqual(command, "sequential_matcher")
        self.assertIn("--SequentialMatching.quadratic_overlap", args)
        self.assertEqual(report["matching"], "sequential")

    def test_existing_or_overlapping_output_is_rejected(self):
        for output in (self.scan, self.scan / "output", self.root):
            with self.subTest(output=output), self.assertRaises(ValueError):
                colmap_sfm.reconstruct(self.scan, output)
        self.output.mkdir()
        marker = self.output / "existing"
        marker.write_text("keep")
        with self.assertRaises(ValueError):
            self.run_fake()
        self.assertEqual(marker.read_text(), "keep")

    def test_manifest_contract_and_legacy_fallback(self):
        request = {"version": 1, "method": "colmap", "inputPoses": "poses_refined.jsonl"}
        path = self.scan / "sfm-request.json"
        path.write_text(json.dumps(request))
        self.assertEqual(colmap_sfm.load_frames(self.scan), self.records)
        request["inputPoses"] = "../outside.jsonl"
        path.write_text(json.dumps(request))
        with self.assertRaises(ValueError):
            colmap_sfm.load_frames(self.scan)
        path.unlink()
        (self.scan / "poses_refined.jsonl").rename(self.scan / "poses.jsonl")
        self.assertEqual(colmap_sfm.load_frames(self.scan), self.records)

    def test_unsafe_or_duplicate_names_are_rejected(self):
        for name in ("../outside.jpg", "a\\b.jpg", "/tmp/a.jpg", "a\nb.jpg", "a\0b.jpg",
                     self.records[1]["imageFile"]):
            with self.subTest(name=name):
                self.records[0]["imageFile"] = name
                self.write_records()
                with self.assertRaises(ValueError):
                    colmap_sfm.load_frames(self.scan)

    def test_symlink_escape_is_rejected(self):
        image = self.scan / "images" / self.records[0]["imageFile"]
        image.unlink()
        outside = self.root / "outside.jpg"
        outside.write_bytes(b"private")
        image.symlink_to(outside)
        with self.assertRaises(ValueError):
            colmap_sfm.load_frames(self.scan)

    def test_request_symlink_escape_is_rejected(self):
        outside = self.root / "outside.json"
        outside.write_text("{}")
        (self.scan / "sfm-request.json").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "request must be inside"):
            colmap_sfm.load_frames(self.scan)

    def test_invalid_intrinsics_are_rejected(self):
        for key, value in (("fx", 0), ("fy", math.nan), ("cx", math.inf),
                           ("width", 0), ("height", 2.5), ("fx", True)):
            original = self.records[0]["intrinsics"][key]
            with self.subTest(key=key, value=value):
                self.records[0]["intrinsics"][key] = value
                self.write_records()
                with self.assertRaises(ValueError):
                    colmap_sfm.load_frames(self.scan)
                self.records[0]["intrinsics"][key] = original

    def test_missing_image_and_too_few_frames_are_rejected(self):
        image = self.scan / "images" / self.records[0]["imageFile"]
        image.unlink()
        with self.assertRaises(ValueError):
            colmap_sfm.load_frames(self.scan)
        self.records = self.records[1:3]
        self.write_records()
        with self.assertRaisesRegex(ValueError, "at least three"):
            colmap_sfm.load_frames(self.scan)

    def test_dimension_mismatch_and_shared_camera_are_rejected(self):
        database = self.root / "test.db"
        self.create_database(database)
        with sqlite3.connect(database) as connection:
            connection.execute("UPDATE cameras SET width = 480 WHERE camera_id = 1")
        with self.assertRaisesRegex(ValueError, "dimensions disagree"):
            colmap_sfm.calibrate_database(database, self.records)
        with sqlite3.connect(database) as connection:
            connection.execute("UPDATE images SET camera_id = 2 WHERE image_id = 1")
        with self.assertRaisesRegex(ValueError, "separate camera"):
            colmap_sfm.calibrate_database(database, self.records)

    def test_cli_option_compatibility(self):
        for prefix in ("SiftExtraction", "FeatureExtraction"):
            with self.subTest(prefix=prefix), patch("colmap_sfm.shutil.which", return_value="/bin/colmap"), \
                    patch("colmap_sfm.subprocess.run",
                          return_value=subprocess.CompletedProcess([], 0, f"--{prefix}.use_gpu arg", "")):
                client = colmap_sfm.Colmap("colmap")
                self.assertEqual(client.option("feature_extractor", "FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"),
                                 "--" + prefix + ".use_gpu")

    def test_missing_colmap_is_actionable(self):
        with patch("colmap_sfm.shutil.which", return_value=None):
            with self.assertRaisesRegex(ValueError, "Install the desktop CLI"):
                colmap_sfm.reconstruct(self.scan, self.output)
        self.assertFalse(self.output.exists())

    @unittest.skipUnless(os.environ.get("COLMAP_INTEGRATION") == "1" and shutil.which("colmap"),
                         "Set COLMAP_INTEGRATION=1 with COLMAP installed for real CPU SfM")
    def test_real_colmap_on_synthetic_textured_corner(self):
        # PPM avoids a Python imaging dependency. Two planes with fixed world-space texture,
        # viewed from translated cameras, supply genuine correspondences and parallax.
        width, height, focal = 480, 360, 380

        @functools.lru_cache(maxsize=200_000)
        def noise(x, y, seed):
            h = ((x * 374761393 + y * 668265263 + seed) & 0xFFFFFFFF)
            h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
            return ((h ^ (h >> 16)) & 0xFFFF) / 65535

        def texture(u, v, cell, seed):
            x, y = u / cell, v / cell
            ix, iy = math.floor(x), math.floor(y)
            a, b = x - ix, y - iy
            a, b = a * a * (3 - 2 * a), b * b * (3 - 2 * b)
            top = noise(ix, iy, seed) * (1 - a) + noise(ix + 1, iy, seed) * a
            bottom = noise(ix, iy + 1, seed) * (1 - a) + noise(ix + 1, iy + 1, seed) * a
            return top * (1 - b) + bottom * b

        self.records = []
        for index in range(10):
            name = f"synthetic_{index:03d}.ppm"
            origin = -0.6 + index * 0.12
            pixels = bytearray()
            for y in range(height):
                dy = (y - height / 2) / focal
                for x in range(width):
                    dx = (x - width / 2) / focal
                    distance, u, v, seed = 3.0, origin + 3 * dx, 3 * dy, 1
                    if dy > 0 and 0.9 / dy < distance:
                        distance = 0.9 / dy
                        u, v, seed = origin + distance * dx, distance, 2
                    if dx < 0 and (-1.4 - origin) / dx < distance:
                        distance = (-1.4 - origin) / dx
                        u, v, seed = distance, distance * dy, 3
                    value = int(255 * (0.1 + 0.6 * texture(u, v, 0.035, seed)
                                       + 0.25 * texture(u, v, 0.012, seed + 7)))
                    pixels.extend((value, value, value))
            (self.scan / "images" / name).write_bytes(f"P6\n{width} {height}\n255\n".encode() + pixels)
            self.records.append({"imageFile": name,
                                 "intrinsics": {"fx": focal, "fy": focal, "cx": width / 2,
                                                "cy": height / 2, "width": width, "height": height}})
        self.write_records()
        before = self.snapshot()
        for matching in ("exhaustive", "sequential"):
            with self.subTest(matching=matching):
                report = colmap_sfm.reconstruct(self.scan, self.output / matching,
                                               matching=matching, threads=2)
                self.assertGreaterEqual(len(report["registeredImages"]), 8)
                self.assertGreater(report["points"], 100)
                self.assertLess(report["meanReprojectionErrorPx"], 1)
                self.assertEqual(before, self.snapshot())


if __name__ == "__main__":
    unittest.main()
