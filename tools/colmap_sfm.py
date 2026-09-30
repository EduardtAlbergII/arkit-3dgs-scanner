#!/usr/bin/env python3
"""Reconstruct an exported scan with desktop COLMAP, without trusting ARKit poses.

Requires a COLMAP CLI installation; Python uses only the standard library.
The input is read-only. Output is published only after reconstruction succeeds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import struct
import subprocess
import tempfile


def load_frames(scan: Path) -> list[dict]:
    request = scan / "sfm-request.json"
    if request.exists():
        if request.resolve().parent != scan:
            raise ValueError("SfM request must be inside the scan directory.")
        settings = json.loads(request.read_text())
        if (settings.get("version") != 1 or settings.get("method") != "colmap"
                or settings.get("inputPoses") != "poses_refined.jsonl"):
            raise ValueError("Unsupported sfm-request.json; export the scan again.")
        poses = scan / "poses_refined.jsonl"
    else:
        poses = next((scan / name for name in
                      ("poses_refined.jsonl", "review-poses.jsonl", "poses.jsonl")
                      if (scan / name).is_file()), scan / "poses.jsonl")
    if poses.resolve().parent != scan:
        raise ValueError("Pose records must be inside the scan directory.")
    records = [json.loads(line) for line in poses.read_text().splitlines() if line.strip()]
    names: set[str] = set()
    for record in records:
        name = record["imageFile"]
        if (not isinstance(name, str) or not name or name in names
                or any(c in name for c in ("/", "\\", "\0", "\n", "\r"))
                or name in (".", "..")):
            raise ValueError(f"Invalid or duplicate image filename: {name!r}")
        names.add(name)
        image = scan / "images" / name
        if not image.is_file() or image.resolve().parent != scan / "images":
            raise ValueError(f"Missing image or image outside the scan: {name}")
        intrinsics = record["intrinsics"]
        for key in ("width", "height"):
            value = intrinsics[key]
            if type(value) is not int or value <= 0:
                raise ValueError(f"Invalid {key} for {name}")
        for key in ("fx", "fy", "cx", "cy"):
            value = intrinsics[key]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or (key in ("fx", "fy") and value <= 0)):
                raise ValueError(f"Invalid {key} for {name}")
    if len(records) < 3:
        raise ValueError("COLMAP reconstruction needs at least three selected photos.")
    return records


class Colmap:
    def __init__(self, executable: str):
        self.executable = shutil.which(executable)
        if not self.executable:
            raise ValueError("COLMAP was not found. Install the desktop CLI or pass --colmap.")
        self.environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        self.help: dict[str, str] = {}

    def option(self, command: str, *names: str) -> str:
        """COLMAP 3.13+ renamed the SIFT execution options to Feature*."""
        if command not in self.help:
            result = subprocess.run([self.executable, command, "-h"],
                                    env=self.environment, capture_output=True, text=True, check=True)
            self.help[command] = result.stdout + result.stderr
        for name in names:
            if re.search(r"--" + re.escape(name) + r"(?=\s|=|$)", self.help[command]):
                return "--" + name
        raise ValueError(f"Unsupported COLMAP CLI: {command} needs one of {names}.")

    def run(self, command: str, *arguments: object) -> None:
        print(f"COLMAP: {command}", flush=True)
        subprocess.run([self.executable, command, *map(str, arguments)],
                       env=self.environment, check=True)


def calibrate_database(database: Path, records: list[dict]) -> None:
    """Use each photo's measured calibration, including autofocus breathing."""
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT images.name, images.camera_id, cameras.width, cameras.height "
            "FROM images JOIN cameras ON images.camera_id = cameras.camera_id"
        ).fetchall()
        by_name = {row[0]: row[1:] for row in rows}
        if set(by_name) != {record["imageFile"] for record in records}:
            raise ValueError("COLMAP did not extract all selected photos.")
        if len({row[1] for row in rows}) != len(rows):
            raise ValueError("COLMAP must create a separate camera for every photo.")
        for record in records:
            name, k = record["imageFile"], record["intrinsics"]
            camera_id, width, height = by_name[name]
            if (width, height) != (k["width"], k["height"]):
                raise ValueError(f"Image/calibration dimensions disagree for {name}. "
                                 "Use original sensor-oriented photos, not rotated/resized copies.")
            params = struct.pack("<4d", *(k[key] for key in ("fx", "fy", "cx", "cy")))
            connection.execute("UPDATE cameras SET model = 1, params = ?, prior_focal_length = 1 "
                               "WHERE camera_id = ?", (params, camera_id))


def model_stats(model: Path) -> dict:
    """Read COLMAP's text model after conversion (image names may contain spaces)."""
    names = []
    image_lines = iter((model / "images.txt").read_text().splitlines())
    for line in image_lines:
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split(maxsplit=9)
        if len(fields) != 10:
            raise ValueError("Invalid COLMAP image record.")
        if not all(math.isfinite(float(value)) for value in fields[1:8]):
            raise ValueError("COLMAP returned a non-finite pose.")
        names.append(fields[9])
        # Every image has a second line of POINTS2D, which may be empty.
        if next(image_lines, None) is None:
            raise ValueError("Missing COLMAP image observations.")
    errors = []
    for line in (model / "points3D.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        coordinates = [float(value) for value in fields[1:4]]
        error = float(fields[7])
        if not all(math.isfinite(value) for value in coordinates + [error]) or error < 0:
            raise ValueError("COLMAP returned invalid sparse points.")
        errors.append(error)
    if len(set(names)) != len(names):
        raise ValueError("COLMAP returned duplicate registered images.")
    return {"registeredImages": sorted(names), "points": len(errors),
            "meanReprojectionErrorPx": sum(errors) / len(errors) if errors else None}


def reconstruct(scan: Path, output: Path, *, executable: str = "colmap",
                matching: str = "exhaustive", min_registered_fraction: float = 0.8,
                threads: int = 4) -> dict:
    scan, output = scan.resolve(), output.resolve()
    if (output.exists() or output == scan or output.is_relative_to(scan)
            or scan.is_relative_to(output)):
        raise ValueError("Output must be a new directory outside the input scan.")
    if matching not in ("exhaustive", "sequential"):
        raise ValueError("Matching must be exhaustive or sequential.")
    if not 0 < min_registered_fraction <= 1 or threads < 1:
        raise ValueError("Registration fraction must be in (0, 1]; threads must be positive.")
    records = load_frames(scan)
    colmap = Colmap(executable)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".colmap-sfm-", dir=output.parent) as temporary:
        work = Path(temporary)
        images = work / "input-images"
        images.mkdir()
        # Only selected images enter SfM. Copy rather than hard-link so external tools cannot
        # modify originals. Depth and ARKit point clouds never enter the new coordinate frame.
        for record in records:
            shutil.copyfile(scan / "images" / record["imageFile"], images / record["imageFile"])
        database = work / "database.db"
        extraction_gpu = colmap.option("feature_extractor", "FeatureExtraction.use_gpu", "SiftExtraction.use_gpu")
        extraction_threads = colmap.option("feature_extractor", "FeatureExtraction.num_threads", "SiftExtraction.num_threads")
        colmap.run("feature_extractor", "--database_path", database, "--image_path", images,
                   "--ImageReader.camera_model", "PINHOLE", "--ImageReader.single_camera_per_image", 1,
                   extraction_gpu, 0, extraction_threads, threads)
        calibrate_database(database, records)
        matcher = matching + "_matcher"
        matching_gpu = colmap.option(matcher, "FeatureMatching.use_gpu", "SiftMatching.use_gpu")
        matching_threads = colmap.option(matcher, "FeatureMatching.num_threads", "SiftMatching.num_threads")
        guided = colmap.option(matcher, "FeatureMatching.guided_matching", "SiftMatching.guided_matching")
        match_options: list[object] = []
        if matching == "sequential":
            match_options = ["--SequentialMatching.overlap", 20, "--SequentialMatching.quadratic_overlap", 1]
        colmap.run(matcher, "--database_path", database, matching_gpu, 0,
                   matching_threads, threads, guided, 1, *match_options)
        sparse = work / "models"
        sparse.mkdir()
        colmap.run("mapper", "--database_path", database, "--image_path", images,
                   "--output_path", sparse, "--Mapper.num_threads", threads,
                   "--Mapper.ba_refine_focal_length", 0, "--Mapper.ba_refine_principal_point", 0,
                   "--Mapper.ba_refine_extra_params", 0)
        candidates = []
        selected_names = {record["imageFile"] for record in records}
        for model in sorted(sparse.iterdir()):
            if not model.is_dir() or not (model / "images.bin").is_file():
                continue
            colmap.run("model_converter", "--input_path", model, "--output_path", model,
                       "--output_type", "TXT")
            stats = model_stats(model)
            if not set(stats["registeredImages"]).issubset(selected_names):
                raise ValueError("COLMAP registered an image outside the selected input.")
            if stats["points"] > 0:
                candidates.append((model, stats))
        if not candidates:
            raise ValueError("COLMAP could not reconstruct a connected scene with sparse points. "
                             "Capture more overlapping, sharp photos of textured surfaces.")
        best, stats = max(candidates, key=lambda item: (len(item[1]["registeredImages"]),
                                                       item[1]["points"]))
        registered = len(stats["registeredImages"])
        if registered < 3 or registered / len(records) < min_registered_fraction:
            raise ValueError(f"Largest connected model registered {registered}/{len(records)} photos; "
                             "output not published. Add overlap or explicitly lower "
                             "--min-registered-fraction after reviewing missing coverage.")
        dataset = work / "dataset"
        colmap.run("image_undistorter", "--image_path", images, "--input_path", best,
                   "--output_path", dataset, "--output_type", "COLMAP")
        # image_undistorter writes sparse directly; 3DGS loaders expect sparse/0.
        exported_sparse = dataset / "sparse"
        model_zero = exported_sparse / "0"
        model_zero.mkdir()
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            source = exported_sparse / name
            if not source.is_file():
                raise ValueError(f"COLMAP did not export {name}.")
            source.rename(model_zero / name)
        # Newer COLMAP models also carry rig/frame associations.
        for name in ("rigs.bin", "frames.bin"):
            source = exported_sparse / name
            if source.is_file():
                source.rename(model_zero / name)
        for name in stats["registeredImages"]:
            if not (dataset / "images" / name).is_file():
                raise ValueError(f"COLMAP did not export registered image {name}.")
        report = {
            "version": 1, "method": "colmap", "matching": matching,
            "inputImages": len(records), **stats,
            "unregisteredImages": sorted(selected_names - set(stats["registeredImages"])),
            "connectedModels": len(candidates), "metricScale": False,
            "coordinateFrame": "colmap-independent",
            "calibration": "per-image-arkit-pinhole-fixed",
        }
        (dataset / "sfm-report.json").write_text(json.dumps(report, indent=2) + "\n")
        # Never overwrite an earlier result, including one created while COLMAP ran.
        if output.exists():
            raise ValueError("Output appeared during reconstruction; refusing to overwrite it.")
        dataset.rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", type=Path, help="Extracted scan directory (not a ZIP)")
    parser.add_argument("-o", "--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--colmap", default="colmap", help="COLMAP executable")
    parser.add_argument("--matching", choices=("exhaustive", "sequential"), default="exhaustive",
                        help="Exhaustive finds revisits; sequential is cheaper for long ordered scans")
    parser.add_argument("--min-registered-fraction", type=float, default=0.8)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    try:
        report = reconstruct(args.scan, args.out, executable=args.colmap, matching=args.matching,
                             min_registered_fraction=args.min_registered_fraction, threads=args.threads)
    except (OSError, ValueError, KeyError, TypeError, IndexError, sqlite3.Error,
            subprocess.CalledProcessError) as error:
        parser.exit(1, f"COLMAP reconstruction failed: {error}\n")
    print(f"Reconstructed {len(report['registeredImages'])}/{report['inputImages']} photos, "
          f"{report['points']} points. Output: {args.out}")
    print("Independent COLMAP coordinates; metric scale is unknown. Do not mix with ARKit depth/points.")


if __name__ == "__main__":
    main()
