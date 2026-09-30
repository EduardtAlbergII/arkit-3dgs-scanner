# Training-format export from scan history

**English** | [繁體中文](HISTORY_TRAINING_EXPORT.zh-TW.md)

Previously, stopping saved only the preview point cloud, playback poses, and summary. Live export created COLMAP files, but history only zipped the folder or shared an old archive, leaving some datasets without `sparse/0`.

Both export entry points now call `ExportManager.writeTrainingDataset`. History prefers review poses, then refined or original poses, and incorporates frames added by resumed capture. Geometry-eligible `.keep` frames are further filtered by the RGB selection report when present. At most 250,000 saved preview points are loaded; older scans without a point cloud use bounded preview reconstruction. Original photos and depth remain intact.

The default ARKit export includes `images/`, `sparse/0/cameras.bin`, `images.bin`, `points3D.bin`, `points.ply`, and `poses_refined.jsonl`. COLMAP poses and points receive the same coordinate conversion. `points.ply` keeps ARKit world coordinates; COLMAP trainers should initialize from `sparse/0/points3D.bin`. Writing this format alone does **not** run COLMAP SfM.

Export validates IDs, pose dimensions and finite values, intrinsics, filenames, and image existence. No valid training frames is an explicit error. Failure preserves the previous ZIP. Entering history details clears the directly shareable cached URL so a complete training package is prepared again.

An empty cloud still produces a valid zero-point `points3D.bin` and empty PLY, replacing stale data. Some trainers need additional initialization. `gaussians.ply` is a trained model, not a required input format. On-device training keeps its checkpoint and model in `gaussian-training/`, which the dataset ZIP leaves out (it zips a hard-linked mirror without that folder, so no media is copied); the model has its own `scan_…-3dgs.zip`. Old scan models remain available for export or deletion.

Archives are shared through the system share sheet with the file itself. SwiftUI `ShareLink(item: URL)` handed apps a file link that only AirDrop and Files accepted; LINE and Teams failed. Very large archives can still exceed a receiving app's own size limit.

## Independent COLMAP reconstruction

The post-capture and history export controls offer **ARKit** (the existing ready-to-train seed model) or **COLMAP (computer)**. The selection persists between screens and launches; changing it clears the share-ready archive so the next export uses the selected method. Capture, preview, measurements, and on-device 3DGS training still use the existing ARKit pipeline.

Use COLMAP when local ARKit-guided refinement cannot recover sufficient multi-view alignment:

1. Choose **COLMAP (computer)**, export, and transfer the ZIP to your computer yourself. The app does not upload anything.
2. Extract the ZIP. This package intentionally omits `sparse/` and `points.ply`: it is an **SfM input**, not an already reconstructed training dataset. It includes `sfm-request.json`, selected-frame records in `poses_refined.jsonl`, and original photos/depth. The phone's saved ARKit model remains intact.
3. Install the desktop [COLMAP CLI](https://colmap.github.io/install.html) and Python 3.10+. Run this repository's tool (no additional Python packages):

   ```sh
   python3 /path/to/repository/tools/colmap_sfm.py /path/to/extracted_scan \
     -o /path/to/new_colmap_dataset
   ```

4. Train using the **new** dataset's `images/` and `sparse/0`, not the input ZIP. Inspect `sfm-report.json` for registered/missing photos, connected model count, point count, and mean reprojection error.

The tool copies only selected photos into temporary storage, extracts SIFT features, performs geometrically verified and guided matching, and runs incremental mapping with bundle adjustment. ARKit **poses and point clouds are not used** as priors or initialization. Each photo retains its own measured PINHOLE intrinsics, including autofocus changes; focal length, principal point, and extra parameters are held fixed during mapping. Pixel dimensions must match calibration; rotated or resized copies are rejected. Undistortion produces the paired images and sparse model for training.

In the COLMAP input package, `poses_refined.jsonl` may contain only photo IDs, filenames, and intrinsics; it does not claim to contain reconstructed poses. The app refreshes this selection from current records, including resumed capture, without overwriting the saved scan's sidecars. Use `colmap_sfm.py`, not `arkit2gs.py`, for this package.

Exhaustive matching is the default so distant revisits can connect; it is quadratic in photo count and can be expensive. `--matching sequential` uses an overlap of 20 with additional exponentially spaced pairs, but can miss long loops. `--threads N` controls extraction, matching, and mapping workers. Processing defaults to CPU (no CUDA required). `--colmap /path/to/colmap` chooses an installation; the tool detects both older SIFT and newer Feature execution option names.

Only the largest connected model is published, and only if it has points and registers at least three photos and 80% of the selected input. Otherwise the command fails without publishing a dataset or silently returning ARKit geometry. `--min-registered-fraction 0.6`, for example, explicitly permits lower coverage; review the missing-image list before lowering this gate. Existing output directories are never overwritten; failed jobs clean their temporary data and leave the input unchanged.

**Limits:** this is desktop SfM, not an iOS COLMAP integration or automatic re-import into the app. The new model has arbitrary scale/orientation and is not aligned to ARKit. Do not attach original LiDAR depth, metric measurements, or ARKit PLY points to it without a separately validated alignment/refusion. The original depth is retained only in the input package, not copied into the output dataset. Fixed PINHOLE calibration does not model rolling shutter or unmodeled distortion. Low texture, repeated patterns, motion blur, moving objects, or disconnected views can still fail; neither registration coverage nor a low training reprojection error proves metric accuracy or improved 3DGS quality.

### COLMAP regression checks

```sh
python3 tools/test_colmap_sfm.py
COLMAP_INTEGRATION=1 python3 tools/test_colmap_sfm.py
```

The first command tests the workflow with a simulated CLI, per-image calibration, connected-model selection, failures, path validation, and input preservation. The optional integration case renders a textured corner and runs real CPU COLMAP through reconstruction and dataset publication. Synthetic evidence is not a measurement of iPhone accuracy, runtime, or memory; compare real scans and independent measurements before claiming accuracy gains.

## Dataset layout (ARKit mode)

Use `sparse/0/images.bin` as the list of training images; `images/` keeps every original photo.

```text
scan_…/
├── images/                    # Original sensor-oriented JPEGs
├── depth/                     # LiDAR depth and confidence, when captured
├── sparse/0/
│   ├── cameras.bin            # Per-frame calibration
│   ├── images.bin             # Selected images and world-to-camera poses
│   └── points3D.bin           # Initialization point cloud
├── points.ply                 # Point cloud in ARKit world coordinates
├── poses.jsonl                # Original capture poses
├── poses_refined.jsonl        # Selected, processed poses
├── capture-meta.json          # Device and capture mode
├── review.ply                 # Saved preview point cloud
├── review-poses.jsonl         # Preview and playback poses
├── scan-summary.json          # Frame and point counts
├── training-selection.json    # Image selection and recapture information
├── pose-refinement.json       # Pose validation report, when refinement runs
└── refusion-progress.json     # Fusion timing and resource report, when fusion runs
```

Floor-plan, world-map, reconstruction, and performance files are added when those features run.

**Coordinates:** COLMAP cameras and `points3D.bin` are rotated together by 180° around world X. `points.ply`, `review.ply`, and the JSONL poses keep ARKit world coordinates. Do not mix the two frames without converting. See [coordinate conventions](COORDINATES.md).

## Regression validation

```sh
swiftc arkit-3dgs-scanner/Capture/TrainingFrameSelector.swift -O -module-cache-path /tmp/fable-swift-cache \
  arkit-3dgs-scanner/Capture/{Localization,Models,BlurFilter,CaptureConfig,DepthSampleFilter,RefusionEngine,ExportManager}.swift \
  arkit-3dgs-scanner/History/ScanLibrary.swift tools/test_history_training_export.swift \
  -o /tmp/fable-history-export-test
/tmp/fable-history-export-test
```

The 14 export checks cover preview-only legacy scans, ZIP replacement, counts/calibration/poses/coordinates in all three binaries, quality selection, corrected-pose priority, preserved raw files, actual ZIP contents, missing images, malformed poses, failed export preserving an old archive, and empty clouds. History and capture/ZIP regressions provide additional coverage. Device builds do not replace sensor testing.

## Repairing a downloaded legacy scan

`tools/prepare_training_export.swift` follows the same history path, writes training files into the supplied scan directory, and creates a sibling ZIP. Copy the original directory first if an unchanged backup is needed.

```sh
swiftc arkit-3dgs-scanner/Capture/TrainingFrameSelector.swift -O -module-cache-path /tmp/fable-swift-cache \
  arkit-3dgs-scanner/Capture/{Localization,Models,BlurFilter,CaptureConfig,DepthSampleFilter,RefusionEngine,ExportManager}.swift \
  arkit-3dgs-scanner/History/ScanLibrary.swift tools/prepare_training_export.swift \
  -o /tmp/fable-prepare-training-export
/tmp/fable-prepare-training-export /path/to/scan_directory
```

## Capture metadata filename

New captures use `capture-meta.json` to avoid trainer detection of unrelated `meta.json` formats. Legacy metadata remains readable and is renamed before preparing or packaging exports, preserving its bytes and unknown fields. When both names exist, the new file takes priority and the old file is preserved as `capture-meta-legacy-UUID.json`.

Already-shared ZIPs do not change automatically. Export again with the updated app, or rename `meta.json` inside an extracted legacy scan. `tools/arkit2gs.py` reads both names.
