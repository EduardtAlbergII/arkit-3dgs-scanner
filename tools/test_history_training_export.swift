import Foundation

@main struct HistoryTrainingExportTests {
    static func main() async {
        do { try await run() }
        catch { FileHandle.standardError.write(Data("FAILED: \(error)\n".utf8)); exit(1) }
    }
    static func run() async throws {
        let fm = FileManager.default
        let root = URL(fileURLWithPath: fm.currentDirectoryPath)
            .appendingPathComponent(".history-training-export-" + UUID().uuidString)
        defer { try? fm.removeItem(at: root) }
        let scan = root.appendingPathComponent("scan_fixture")
        try fm.createDirectory(at: scan.appendingPathComponent("images"), withIntermediateDirectories: true)
        let library = ScanLibrary(root: root)
        var checks = 0
        func check(_ condition: Bool, _ message: String) {
            guard condition else { FileHandle.standardError.write(Data("FAIL: \(message)\n".utf8)); exit(1) }
            checks += 1; print("PASS: \(message)")
        }
        func zipContents(_ archive: URL, path: String? = nil) throws -> Data {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
            process.arguments = path.map { ["-p", archive.path, "scan_fixture/" + $0] } ?? ["-Z1", archive.path]
            let pipe = Pipe(); process.standardOutput = pipe
            try process.run()
            let result = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            check(process.terminationStatus == 0, "ZIP contents are readable")
            return result
        }
        func snapshot() throws -> [String: Data] {
            var result: [String: Data] = [:]
            let files = fm.enumerator(at: scan, includingPropertiesForKeys: [.isRegularFileKey])!
            for case let file as URL in files {
                if try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true {
                    result[String(file.path.dropFirst(scan.path.count + 1))] = try Data(contentsOf: file)
                }
            }
            return result
        }
        func record(_ id: Int) -> FrameRecord {
            FrameRecord(id: id, timestamp: Double(id), transform: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
                        intrinsics: CameraIntrinsics(fx: 100, fy: 101, cx: 50, cy: 40, width: 100, height: 80),
                        exposureDuration: 0.005, exposureOffsetEV: 0, estimatedBlurPx: 2, imageFile: "frame_\(id).jpg")
        }
        let raw = [record(1), record(2), record(3), record(4), record(5)]
        for r in raw { try Data("image \(r.id)".utf8).write(to: scan.appendingPathComponent("images/" + r.imageFile)) }
        try ExportManager.writeRefinedPoses(raw, to: scan.appendingPathComponent("poses.jsonl"))
        var reviewed = raw
        reviewed[0].transform[3] = 1.25
        reviewed[1].blurVerdict = .demote
        reviewed[2].blurVerdict = .drop
        let points = [CloudPoint(x: 0.1, y: 0.2, z: -2, r: 100, g: 120, b: 140)]
        try await library.saveReview(directory: scan, points: points, records: reviewed)
        let entry = try await library.entries()[0]
        let stale = try ExportManager.makeArchive(of: scan)
        let staleBytes = try Data(contentsOf: stale)
        check(!fm.fileExists(atPath: scan.appendingPathComponent("sparse/0").path), "fixture reproduces preview-only export without COLMAP")
        let reviewOnly = try snapshot()
        var invalidPoseRecords = reviewed
        invalidPoseRecords[0].transform = [.nan]
        _ = try ExportManager.makeArchive(of: scan, method: .colmap, records: invalidPoseRecords)
        check((try snapshot()) == reviewOnly, "nonfinite ARKit poses do not block isolated desktop photo export")
        let desktopReviewArchive = try await library.archive(entry, method: .colmap)
        let desktopReviewPoses = try zipContents(desktopReviewArchive, path: "poses_refined.jsonl")
        let desktopReviewInputs = try String(decoding: desktopReviewPoses, as: UTF8.self)
            .split(whereSeparator: \.isNewline)
            .map { try JSONSerialization.jsonObject(with: Data($0.utf8)) as! [String: Any] }
        check(desktopReviewInputs.count == 3 && desktopReviewInputs[0]["imageFile"] as? String == "frame_1.jpg"
              && desktopReviewInputs[0]["transform"] == nil,
              "desktop export creates missing photo calibration inputs in staging without requiring ARKit poses")
        check((try snapshot()) == reviewOnly, "desktop export leaves review-only source files unchanged")
        let archive = try await library.archive(entry)
        let repairedBytes = try Data(contentsOf: archive)
        check(archive.standardizedFileURL.path == stale.standardizedFileURL.path && repairedBytes != staleBytes, "history replaces the incomplete existing ZIP")
        let sparse = scan.appendingPathComponent("sparse/0")
        let cameras = try Data(contentsOf: sparse.appendingPathComponent("cameras.bin"))
        let images = try Data(contentsOf: sparse.appendingPathComponent("images.bin"))
        let seeds = try Data(contentsOf: sparse.appendingPathComponent("points3D.bin"))
        func uint64(_ data: Data, _ offset: Int = 0) -> UInt64 { data.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt64.self)) } }
        func number(_ data: Data, _ offset: Int) -> Double { Double(bitPattern: uint64(data, offset)) }
        check(uint64(cameras) == 3 && uint64(images) == 3, "only keep frames enter cameras.bin and images.bin")
        check(cameras.count == 176 && number(cameras, 32) == 100, "camera binary contains PINHOLE dimensions and intrinsics")
        check(abs(number(images, 44) + 1.25) < 1e-9, "export uses corrected review poses instead of raw ARKit positions")
        check(uint64(seeds) == 1 && abs(number(seeds, 32) - 2) < 1e-6, "point coordinates use the same world flip as the cameras")
        let exported = ScanLibrary.readRecords(scan.appendingPathComponent("poses_refined.jsonl"))
        check(exported.count == 3 && exported[0].transform[3] == 1.25, "refined sidecar matches training cameras")
        check(ScanLibrary.readRecords(scan.appendingPathComponent("poses.jsonl")).count == 5, "raw poses and excluded source photos remain preserved")
        let unzip = Process(); unzip.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        unzip.arguments = ["-Z1", archive.path]
        let pipe = Pipe(); unzip.standardOutput = pipe
        try unzip.run(); let listing = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self); unzip.waitUntilExit()
        check(unzip.terminationStatus == 0 && ["cameras.bin", "images.bin", "points3D.bin", "points.ply", "poses_refined.jsonl", "frame_1.jpg"].allSatisfy { listing.contains($0) }, "actual ZIP includes the complete dataset and images")
        try fm.createDirectory(at: scan.appendingPathComponent("depth"), withIntermediateDirectories: true)
        try Data([0, 1, 2, 255]).write(to: scan.appendingPathComponent("depth/frame_1.bin"))
        try Data("{ \"custom\": \"metadata bytes\" }\n".utf8).write(to: scan.appendingPathComponent("capture-meta.json"))
        try Data("{ \"legacy\": true }\n".utf8).write(to: scan.appendingPathComponent("meta.json"))
        try fm.createDirectory(at: scan.appendingPathComponent("gaussian-training"), withIntermediateDirectories: true)
        try Data("checkpoint must remain local".utf8).write(to: scan.appendingPathComponent("gaussian-training/checkpoint.bin"))
        try Data("{\"method\":\"stale\"}".utf8).write(to: scan.appendingPathComponent("sfm-request.json"))
        try Data("stale instructions".utf8).write(to: scan.appendingPathComponent("COLMAP-INSTRUCTIONS.txt"))
        let originalFiles = try snapshot()
        let reusedArchive = try ExportManager.makeArchive(of: scan, method: .colmap)
        for path in ["poses_refined.jsonl", "training-selection.json"] {
            check((try zipContents(reusedArchive, path: path)) == originalFiles[path],
                  "without supplied records desktop export preserves existing sidecar bytes: " + path)
        }
        try Data("new resumed image".utf8).write(to: scan.appendingPathComponent("images/frame_6.jpg"))
        try ExportManager.writeRefinedPoses(raw + [record(6)], to: scan.appendingPathComponent("poses.jsonl"))
        let resumedFiles = try snapshot()
        let resumedArchive = try await library.archive(entry, method: .colmap)
        let resumedInputs = try String(decoding: zipContents(resumedArchive, path: "poses_refined.jsonl"), as: UTF8.self)
            .split(whereSeparator: \.isNewline)
            .map { try JSONSerialization.jsonObject(with: Data($0.utf8)) as! [String: Any] }
        check(Set(resumedInputs.compactMap { $0["imageFile"] as? String })
              == Set(["frame_1.jpg", "frame_4.jpg", "frame_5.jpg", "frame_6.jpg"]),
              "history regenerates stale refined inputs with resumed frames")
        var latestRecords = reviewed + [record(6)]
        latestRecords[0].intrinsics.fx = 222
        let latestArchive = try ExportManager.makeArchive(of: scan, method: .colmap, records: latestRecords)
        let latestInputs = try String(decoding: zipContents(latestArchive, path: "poses_refined.jsonl"), as: UTF8.self)
            .split(whereSeparator: \.isNewline)
            .map { try JSONSerialization.jsonObject(with: Data($0.utf8)) as! [String: Any] }
        check((latestInputs[0]["intrinsics"] as? [String: Any])?["fx"] as? Double == 222,
              "explicit live records replace stale refined calibration in the mirror")
        let latestSelection = try JSONSerialization.jsonObject(
            with: zipContents(latestArchive, path: "training-selection.json")) as! [String: Any]
        check(latestSelection["selectedIDs"] as? [Int] == [1, 4, 5, 6],
              "regenerated selection matches the current desktop input records")
        check((try snapshot()) == resumedFiles, "replacing hard-linked sidecars preserves every source byte")
        try originalFiles["poses.jsonl"]!.write(to: scan.appendingPathComponent("poses.jsonl"))
        try fm.removeItem(at: scan.appendingPathComponent("images/frame_6.jpg"))
        let desktopArchive = try await library.archive(entry, method: .colmap)
        check(desktopArchive.standardizedFileURL == archive.standardizedFileURL, "both export methods replace the same archive destination")
        let desktopListing = String(decoding: try zipContents(desktopArchive), as: UTF8.self)
        check(!desktopListing.contains("scan_fixture/sparse/") && !desktopListing.contains("scan_fixture/points.ply")
              && !desktopListing.contains("scan_fixture/gaussian-training/"),
              "desktop archive excludes ARKit sparse seeds and on-device training artifacts")
        let request = try JSONSerialization.jsonObject(with: zipContents(desktopArchive, path: "sfm-request.json")) as! [String: Any]
        check(request["version"] as? Int == 1 && request["method"] as? String == "colmap"
              && request["inputPoses"] as? String == "poses_refined.jsonl", "desktop archive declares the versioned SfM request")
        for path in ["images/frame_1.jpg", "images/frame_2.jpg", "images/frame_3.jpg",
                     "images/frame_4.jpg", "images/frame_5.jpg", "depth/frame_1.bin", "capture-meta.json"] {
            check((try zipContents(desktopArchive, path: path)) == originalFiles[path], "desktop archive preserves bytes: " + path)
        }
        let legacyPath = desktopListing.split(whereSeparator: \.isNewline)
            .map(String.init).first { $0.contains("/capture-meta-legacy-") }!
        check((try zipContents(desktopArchive, path: URL(fileURLWithPath: legacyPath).lastPathComponent)) == originalFiles["meta.json"]
              && !desktopListing.contains("scan_fixture/meta.json"),
              "metadata migration uses standard names in the mirror while preserving both versions")
        let instructions = String(decoding: try zipContents(desktopArchive, path: "COLMAP-INSTRUCTIONS.txt"), as: UTF8.self)
        check(instructions.contains("python3 /path/to/repository/tools/colmap_sfm.py /path/to/extracted_scan -o /path/to/new_dataset")
              && instructions.contains("not metric-aligned") && instructions.contains("Do not mix"),
              "desktop instructions explain processing and LiDAR coordinate isolation")
        check((try snapshot()) == originalFiles, "desktop export preserves every original file including existing training data")
        try fm.removeItem(at: scan.appendingPathComponent("poses.jsonl"))
        try fm.removeItem(at: scan.appendingPathComponent("review-poses.jsonl"))
        func photoOnlyInputs(transform: Any? = nil) throws -> Data {
            var data = Data()
            for id in [1, 4, 5] {
                var input: [String: Any] = [
                    "imageFile": "frame_\(id).jpg",
                    "intrinsics": ["fx": 100, "fy": 101, "cx": 50, "cy": 40, "width": 100, "height": 80]
                ]
                input["transform"] = transform
                data.append(try JSONSerialization.data(withJSONObject: input))
                data.append(0x0a)
            }
            return data
        }
        for transform in [nil, "invalid pose" as Any?, [1] as Any?] {
            let poseFreeBytes = try photoOnlyInputs(transform: transform)
            try poseFreeBytes.write(to: scan.appendingPathComponent("poses_refined.jsonl"))
            let poseFreeArchive = try await library.archive(entry, method: .colmap)
            check((try zipContents(poseFreeArchive, path: "poses_refined.jsonl")) == poseFreeBytes,
                  "desktop history export accepts absent or invalid ARKit poses and preserves their bytes")
        }
        try fm.removeItem(at: scan.appendingPathComponent("poses_refined.jsonl"))
        let rawPhotoInputs = try photoOnlyInputs()
        try rawPhotoInputs.write(to: scan.appendingPathComponent("poses.jsonl"))
        let rawPhotoArchive = try await library.archive(entry, method: .colmap)
        check((try zipContents(rawPhotoArchive, path: "poses_refined.jsonl")) == rawPhotoInputs,
              "desktop export can use legacy photo calibration records without decodable ARKit poses")
        for path in ["poses.jsonl", "review-poses.jsonl", "poses_refined.jsonl"] {
            try originalFiles[path]!.write(to: scan.appendingPathComponent(path))
        }
        _ = try await library.archive(entry, method: .colmap)
        check(try fm.contentsOfDirectory(atPath: root.path).allSatisfy { !$0.hasPrefix(".archive-") && !$0.hasSuffix(".partial") },
              "successful desktop export removes isolated staging")
        let validDesktopArchive = try Data(contentsOf: desktopArchive)
        do {
            _ = try ExportManager.makeArchive(of: scan, method: .colmap, records: [record(1), record(4)])
            preconditionFailure("fewer than three selected photos should fail")
        } catch ExportManager.TrainingExportError.insufficientCOLMAPImages {
            checks += 1; print("PASS: desktop export requires at least three selected photos")
        }
        check((try Data(contentsOf: desktopArchive)) == validDesktopArchive, "too few desktop photos preserve the previous ZIP")
        try fm.removeItem(at: scan.appendingPathComponent("images/frame_1.jpg"))
        do { _ = try await library.archive(entry, method: .colmap); preconditionFailure("missing desktop image should fail") }
        catch ExportManager.TrainingExportError.missingImage { checks += 1; print("PASS: missing desktop input prevents export") }
        check((try Data(contentsOf: desktopArchive)) == validDesktopArchive, "failed desktop export preserves the previous ZIP")
        check(try fm.contentsOfDirectory(atPath: root.path).allSatisfy { !$0.hasPrefix(".archive-") && !$0.hasSuffix(".partial") },
              "failed desktop export removes isolated staging")
        try originalFiles["images/frame_1.jpg"]!.write(to: scan.appendingPathComponent("images/frame_1.jpg"))
        let restoredArchive = try await library.archive(entry)
        let restoredListing = String(decoding: try zipContents(restoredArchive), as: UTF8.self)
        check(restoredListing.contains("scan_fixture/sparse/0/cameras.bin")
              && restoredListing.contains("scan_fixture/points.ply")
              && !restoredListing.contains("sfm-request.json")
              && !restoredListing.contains("COLMAP-INSTRUCTIONS.txt")
              && !restoredListing.contains("gaussian-training/"),
              "switching back to default ARKit restores training model without stale desktop request")
        let restoredCameras = try Data(contentsOf: sparse.appendingPathComponent("cameras.bin"))
        let restoredImages = try Data(contentsOf: sparse.appendingPathComponent("images.bin"))
        let restoredSeeds = try Data(contentsOf: sparse.appendingPathComponent("points3D.bin"))
        check(restoredCameras == cameras && restoredImages == images && restoredSeeds == seeds,
              "switching export methods does not change default ARKit training binaries")
        let originalArchive = try Data(contentsOf: archive)
        try fm.removeItem(at: scan.appendingPathComponent("images/frame_1.jpg"))
        do { _ = try await library.archive(entry); preconditionFailure("missing image should fail") }
        catch ExportManager.TrainingExportError.missingImage { checks += 1; print("PASS: missing training photo prevents a misleading successful export") }
        check((try Data(contentsOf: archive)) == originalArchive, "failed export preserves the previous valid ZIP")
        try Data("restored".utf8).write(to: scan.appendingPathComponent("images/frame_1.jpg"))
        var invalid = record(1); invalid.transform = [1]
        do { try ExportManager.writeTrainingDataset(records: [invalid], points: points, to: scan); preconditionFailure("invalid pose should fail") }
        catch ExportManager.TrainingExportError.invalidFrame { checks += 1; print("PASS: malformed pose cannot crash binary pose conversion") }
        do { try ExportManager.writeTrainingDataset(records: [], points: points, to: scan); preconditionFailure("empty model should fail") }
        catch ExportManager.TrainingExportError.noUsableFrames { checks += 1; print("PASS: no usable poses reports a clear error instead of silently omitting sparse") }
        // A saved zero-point scan can still export calibrated images, but never stale seed points.
        try ExportManager.writeTrainingDataset(records: [record(1)], points: [], to: scan)
        let emptySeeds = try Data(contentsOf: sparse.appendingPathComponent("points3D.bin"))
        let emptyPLY = try ScanLibrary.readPLY(scan.appendingPathComponent("points.ply"), limit: 100)
        check(uint64(emptySeeds) == 0 && emptyPLY.isEmpty,
              "zero-point dataset replaces stale PLY and still has a valid binary header")
        print("\(checks) checks passed")
    }
}
