//
//  ExportManager.swift
//  fable — COLMAP binary sparse model 匯出（LichtFeld-Studio / Inria 3DGS 直讀）
//         ＋ PLY 點雲 ＋ 零依賴 zip 打包
//
//  匯出後的掃描資料夾本身就是標準 COLMAP 資料集：
//      scan_x/
//      ├── images/                  ← 訓練影像
//      └── sparse/0/
//          ├── cameras.bin          ← PINHOLE 內參（逐幀一組，吸收連續對焦的內參呼吸）
//          ├── images.bin           ← w2c 姿態（qw qx qy qz + t，OpenCV 相機慣例）
//          └── points3D.bin         ← LiDAR 彩色點雲（3DGS 初始化）
//  二進位編排與 COLMAP scripts/python/read_write_model.py 完全一致。
//

import Foundation
import simd

nonisolated enum ExportMethod: String, CaseIterable, Sendable {
    case arkit
    case colmap

    static let storageKey = "scanExportMethod"

    var title: String {
        switch self {
        case .arkit: return L10n.text("ARKit")
        case .colmap: return L10n.text("COLMAP（電腦）")
        }
    }

    var notice: String {
        switch self {
        case .arkit: return L10n.text("照片、相機姿態與點雲，可在電腦上訓練")
        case .colmap: return L10n.text("需在電腦安裝 COLMAP，以照片重新估算姿態。手機不執行 COLMAP，也不會上傳資料。")
        }
    }
}

nonisolated extension Data {
    /// 以主機端序（iOS/macOS 皆為 little-endian，即 COLMAP 要求的端序）附加原始 bytes
    mutating func appendLE<T>(_ value: T) {
        Swift.withUnsafeBytes(of: value) { append(contentsOf: $0) }
    }
}

nonisolated enum ExportManager {
    /// On-device 3DGS training folder inside a scan (checkpoint and trained model).
    static let gaussianTrainingFolder = "gaussian-training"

    /// 先寫暫存檔；壓縮完整成功才以正式檔名發布，避免分享半個 ZIP。
    static func makeArchive(of directory: URL, method: ExportMethod = .arkit,
                            records: [FrameRecord]? = nil) throws -> URL {
        let parent = directory.deletingLastPathComponent()
        let destination = parent.appendingPathComponent(directory.lastPathComponent + ".zip")
        let temporary = parent.appendingPathComponent(UUID().uuidString + ".partial")
        defer { try? FileManager.default.removeItem(at: temporary) }
        // Replace regenerated sidecars atomically in the hard-linked mirror; never edit
        // existing files through their links.
        let fm = FileManager.default
        var excluded: Set<String> = [gaussianTrainingFolder, "sfm-request.json", "COLMAP-INSTRUCTIONS.txt"]
        if method == .colmap { excluded.formUnion(["sparse", "points.ply"]) }
        if method == .colmap || excluded.contains(where: { fm.fileExists(atPath: directory.appendingPathComponent($0).path) }) {
            let staging = parent.appendingPathComponent(".archive-\(UUID().uuidString)", isDirectory: true)
            let mirror = staging.appendingPathComponent(directory.lastPathComponent, isDirectory: true)
            defer { try? fm.removeItem(at: staging) }
            try mirrorTree(directory, to: mirror, excluding: excluded)
            if method == .colmap { try prepareCOLMAPRequest(in: mirror, records: records) }
            try zipDirectory(mirror, to: temporary)
        } else {
            try zipDirectory(directory, to: temporary)
        }
        if FileManager.default.fileExists(atPath: destination.path) {
            _ = try FileManager.default.replaceItemAt(destination, withItemAt: temporary)
        } else {
            try FileManager.default.moveItem(at: temporary, to: destination)
        }
        return destination
    }

    private struct COLMAPPhotoInput: Encodable {
        let id: Int
        let imageFile: String
        let intrinsics: CameraIntrinsics
    }

    private struct COLMAPImageReference: Decodable {
        let imageFile: String
    }

    private static func prepareCOLMAPRequest(in directory: URL, records: [FrameRecord]?) throws {
        let fm = FileManager.default
        let poses = directory.appendingPathComponent("poses_refined.jsonl")
        if let records {
            let selection = TrainingFrameSelector.select(records,
                evidence: TrainingFrameSelector.evidence(records: records, directory: directory),
                workingDistance: TrainingFrameSelector.workingDistance(records: records, directory: directory))
            let selected = Set(selection.selectedIDs)
            let inputs = records.filter { selected.contains($0.id) }
            guard inputs.count >= 3 else { throw TrainingExportError.insufficientCOLMAPImages }
            // Desktop SfM consumes calibration, not ARKit transforms; even nonfinite ARKit
            // poses must not prevent packaging otherwise usable photos.
            var data = Data()
            for input in inputs {
                data.append(try JSONEncoder().encode(COLMAPPhotoInput(
                    id: input.id, imageFile: input.imageFile, intrinsics: input.intrinsics)))
                data.append(0x0a)
            }
            try data.write(to: poses, options: .atomic)
            let report = directory.appendingPathComponent("training-selection.json")
            try JSONEncoder().encode(selection).write(to: report, options: .atomic)
        }
        if !fm.fileExists(atPath: poses.path) {
            guard let source = ["review-poses.jsonl", "poses.jsonl"]
                .map({ directory.appendingPathComponent($0) })
                .first(where: { fm.fileExists(atPath: $0.path) }) else {
                throw TrainingExportError.noUsableFrames
            }
            try fm.copyItem(at: source, to: poses)
        }
        let inputs = try String(contentsOf: poses, encoding: .utf8).split(whereSeparator: \.isNewline)
            .map { try JSONDecoder().decode(COLMAPImageReference.self, from: Data($0.utf8)) }
        guard inputs.count >= 3 else { throw TrainingExportError.insufficientCOLMAPImages }
        for input in inputs {
            let name = input.imageFile
            guard !name.isEmpty, !name.contains("\0"), URL(fileURLWithPath: name).lastPathComponent == name else {
                throw TrainingExportError.missingImage(name)
            }
            let image = directory.appendingPathComponent("images").appendingPathComponent(name)
            guard (try? image.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true else {
                throw TrainingExportError.missingImage(name)
            }
        }
        let request: [String: Any] = ["version": 1, "method": "colmap", "inputPoses": "poses_refined.jsonl"]
        try JSONSerialization.data(withJSONObject: request, options: [.prettyPrinted, .sortedKeys])
            .write(to: directory.appendingPathComponent("sfm-request.json"), options: .atomic)
        let instructions = """
        Desktop COLMAP required. This archive is not a ready-to-train sparse model.
        Re-estimate camera poses from photos on your computer:
        python3 /path/to/repository/tools/colmap_sfm.py /path/to/extracted_scan -o /path/to/new_dataset
        No COLMAP runs on the phone and no data is uploaded automatically.
        The reconstruction is not metric-aligned with ARKit. Do not mix it with LiDAR points or depth.

        需在電腦安裝 COLMAP；此壓縮檔不是可直接訓練的稀疏模型。
        請在電腦執行上述指令，以照片重新估算姿態。
        手機不執行 COLMAP，也不會自動上傳資料。
        重建結果未與 ARKit 的公尺尺度對齊，不可混用 LiDAR 點雲或深度。
        """
        try Data(instructions.utf8).write(to: directory.appendingPathComponent("COLMAP-INSTRUCTIONS.txt"), options: .atomic)
    }


    /// Recreates `source` under `destination` with hard links (copies where linking fails),
    /// skipping top-level entries named in `excluding`.
    static func mirrorTree(_ source: URL, to destination: URL, excluding: Set<String>) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: destination, withIntermediateDirectories: true)
        for item in try fm.contentsOfDirectory(at: source, includingPropertiesForKeys: [.isDirectoryKey]) {
            if excluding.contains(item.lastPathComponent) { continue }
            let target = destination.appendingPathComponent(item.lastPathComponent)
            if (try item.resourceValues(forKeys: [.isDirectoryKey])).isDirectory == true {
                try mirrorTree(item, to: target, excluding: [])
            } else {
                do { try fm.linkItem(at: item, to: target) } catch { try fm.copyItem(at: item, to: target) }
            }
        }
    }

    enum TrainingExportError: LocalizedError {
        case noUsableFrames, invalidFrame(Int), missingImage(String), invalidPoints, insufficientCOLMAPImages
        var errorDescription: String? {
            switch self {
            case .noUsableFrames: return L10n.text("沒有可用的相機姿態與清晰影像，無法產生 3DGS 訓練資料")
            case .invalidFrame(let id): return L10n.text("第 \(id) 張影像的相機參數不完整，無法產生訓練資料")
            case .missingImage(let name): return L10n.text("找不到訓練影像：\(name)")
            case .invalidPoints: return L10n.text("點雲含有無效座標，無法產生訓練資料")
            case .insufficientCOLMAPImages: return L10n.text("電腦 COLMAP 重建至少需要 3 張可用照片")
            }
        }
    }

    /// Live export and history sharing use the same dataset preparation step.
    /// Validate before replacing any previous training files; raw photos/depth remain intact.
    static func writeTrainingDataset(records: [FrameRecord], points: [CloudPoint],
                                     to directory: URL, flipWorldUp: Bool = true) throws {
        let evidence = TrainingFrameSelector.evidence(records: records, directory: directory)
        let selection = TrainingFrameSelector.select(records, evidence: evidence,
                                                     workingDistance: TrainingFrameSelector.workingDistance(records: records, directory: directory))
        let selectedIDs = Set(selection.selectedIDs)
        let records = records.filter { selectedIDs.contains($0.id) }
        guard !records.isEmpty else { throw TrainingExportError.noUsableFrames }
        var ids = Set<Int>(), names = Set<String>()
        for r in records {
            let k = r.intrinsics
            guard r.id > 0, r.id <= Int(Int32.max), ids.insert(r.id).inserted,
                  r.timestamp.isFinite, r.transform.count == 16, r.transform.allSatisfy(\.isFinite),
                  k.width > 0, k.height > 0, k.fx.isFinite, k.fy.isFinite,
                  k.fx > 0, k.fy > 0, k.cx.isFinite, k.cy.isFinite,
                  !r.imageFile.isEmpty, !r.imageFile.contains("\0"),
                  URL(fileURLWithPath: r.imageFile).lastPathComponent == r.imageFile,
                  names.insert(r.imageFile).inserted else { throw TrainingExportError.invalidFrame(r.id) }
            let pose = colmapPose(fromRowMajorC2WGL: r.transform, flipWorldUp: flipWorldUp)
            guard pose.q.vector.x.isFinite, pose.q.vector.y.isFinite, pose.q.vector.z.isFinite,
                  pose.q.vector.w.isFinite, pose.t.x.isFinite, pose.t.y.isFinite, pose.t.z.isFinite else {
                throw TrainingExportError.invalidFrame(r.id)
            }
            let image = directory.appendingPathComponent("images").appendingPathComponent(r.imageFile)
            guard (try? image.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true else {
                throw TrainingExportError.missingImage(r.imageFile)
            }
        }
        guard points.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }) else {
            throw TrainingExportError.invalidPoints
        }
        try CaptureMetadata.migrateLegacyFile(in: directory)
        try writeColmapSparse(records: records, points: points, to: directory, flipWorldUp: flipWorldUp)
        // Always replace even an empty cloud: a previous export must not leave stale seed points.
        try writePLY(points, to: directory.appendingPathComponent("points.ply"))
        try writeRefinedPoses(records, to: directory.appendingPathComponent("poses_refined.jsonl"))
        try JSONEncoder().encode(selection).write(to: directory.appendingPathComponent("training-selection.json"), options: .atomic)
    }

    // MARK: - COLMAP sparse model

    /// 世界上方向對齊：ARKit 世界為 +Y up，但 COLMAP/3DGS 生態多沿用 OpenCV 相機慣例、
    /// 隱含假設重力 ≈ +Y（-Y 為 up），ARKit 資料直接匯入會上下顛倒。
    /// flipWorldUp = 繞世界 X 軸轉 180°（(x,y,z)→(x,-y,-z)），對齊 COLMAP 慣例（預設開啟）。
    /// 這是剛體變換，同時作用於相機姿態與點雲、不改變重建品質，只轉正顯示方向。
    static func writeColmapSparse(records: [FrameRecord], points: [CloudPoint],
                                  to sessionDir: URL, flipWorldUp: Bool = true) throws {
        guard !records.isEmpty else { throw TrainingExportError.noUsableFrames }
        let sparseDir = sessionDir.appendingPathComponent("sparse/0", isDirectory: true)
        try FileManager.default.createDirectory(at: sparseDir, withIntermediateDirectories: true)
        try writeCamerasBin(records: records, to: sparseDir.appendingPathComponent("cameras.bin"))
        try writeImagesBin(records: records, to: sparseDir.appendingPathComponent("images.bin"),
                           flipWorldUp: flipWorldUp)
        try writePoints3DBin(points: points, to: sparseDir.appendingPathComponent("points3D.bin"),
                             flipWorldUp: flipWorldUp)
    }

    /// ARKit c2w（GL 慣例、row-major 16）→ COLMAP 儲存格式：w2c 的（四元數 wxyz、平移）。
    /// 數學：（可選）先左乘世界翻轉 diag(1,-1,-1,1)；再 c2w_cv = c2w_gl · diag(1,-1,-1)
    ///       右乘翻相機局部 Y/Z 軸；R_w2c = R_cvᵀ，t_w2c = -R_cvᵀ·t（剛體解析逆）。
    static func colmapPose(fromRowMajorC2WGL m0: [Double],
                           flipWorldUp: Bool = false) -> (q: simd_quatd, t: SIMD3<Double>) {
        var m = m0
        if flipWorldUp {          // 世界繞 X 軸 180° = 左乘 diag(1,-1,-1,1) = 負 row 1、row 2
            for i in 4..<12 { m[i] = -m[i] }
        }
        let c0 = SIMD3<Double>(m[0], m[4], m[8])
        let c1 = -SIMD3<Double>(m[1], m[5], m[9])
        let c2 = -SIMD3<Double>(m[2], m[6], m[10])
        let t = SIMD3<Double>(m[3], m[7], m[11])
        let rW2C = simd_double3x3(c0, c1, c2).transpose
        var q = simd_normalize(simd_quatd(rW2C))
        if q.real < 0 { q = simd_quatd(real: -q.real, imag: -q.imag) }   // 正規化 qw ≥ 0
        return (q, -(rW2C * t))
    }

    /// 一張影像一組 PINHOLE 內參（camera_id = image_id）。
    ///
    /// 不用「取中位數做單一相機」：那個做法只在對焦被鎖住時成立，而我們刻意讓對焦保持連續自動
    /// （鎖對焦會把整段掃描凍在起始那一刻的景深裡，離開就糊 —— 見 CameraControls.lockForScan）。
    /// 連續對焦會帶來內參呼吸（focus breathing）：鏡組移動使有效焦長變化，iPhone 主鏡由無限遠
    /// 到近距約 1~2%，在 1920 寬的畫面邊緣就是 10~19 px 的重投影誤差 —— 遠大於可以忽略的程度。
    /// 但這是**可修的**：ARKit 每幀都給了當下的內參，FrameRecord 也早就逐幀存下來了。
    /// COLMAP 格式支援一張影像一組相機，外部讀取器可依 camera_id 取得逐幀內參
    /// （load_colmap.cpp: `cameras.find(img.camId)`），所以逐幀輸出的成本是零。
    /// 於是「失焦模糊」（不可修）被換成「內參變動」（完全吸收）。
    private static func writeCamerasBin(records: [FrameRecord], to url: URL) throws {
        var data = Data(capacity: records.count * 48 + 8)
        data.appendLE(UInt64(records.count))                  // num_cameras
        for r in records {
            data.appendLE(Int32(r.id))                        // camera_id = image_id
            data.appendLE(Int32(1))                           // model_id: PINHOLE
            data.appendLE(UInt64(r.intrinsics.width))
            data.appendLE(UInt64(r.intrinsics.height))
            data.appendLE(r.intrinsics.fx)
            data.appendLE(r.intrinsics.fy)
            data.appendLE(r.intrinsics.cx)
            data.appendLE(r.intrinsics.cy)
        }
        try data.write(to: url, options: [.atomic])
    }

    private static func writeImagesBin(records: [FrameRecord], to url: URL,
                                       flipWorldUp: Bool) throws {
        var data = Data(capacity: records.count * 96 + 8)
        data.appendLE(UInt64(records.count))
        for r in records {
            let (q, t) = colmapPose(fromRowMajorC2WGL: r.transform, flipWorldUp: flipWorldUp)
            data.appendLE(Int32(r.id))
            data.appendLE(q.real)
            data.appendLE(q.imag.x)
            data.appendLE(q.imag.y)
            data.appendLE(q.imag.z)
            data.appendLE(t.x)
            data.appendLE(t.y)
            data.appendLE(t.z)
            data.appendLE(Int32(r.id))                        // camera_id = image_id（逐幀內參）
            data.append(r.imageFile.data(using: .utf8)!)
            data.append(0)                                    // name 結尾 \0
            data.appendLE(UInt64(0))                          // num_points2D（無 SfM 觀測）
        }
        try data.write(to: url, options: [.atomic])
    }

    private static func writePoints3DBin(points: [CloudPoint], to url: URL,
                                         flipWorldUp: Bool) throws {
        var data = Data(capacity: points.count * 43 + 8)
        data.appendLE(UInt64(points.count))
        var pointID: UInt64 = 1
        for p in points {
            data.appendLE(pointID)
            pointID += 1
            data.appendLE(Double(p.x))
            data.appendLE(flipWorldUp ? Double(-p.y) : Double(p.y))    // 與姿態同步翻轉
            data.appendLE(flipWorldUp ? Double(-p.z) : Double(p.z))
            data.append(p.r)
            data.append(p.g)
            data.append(p.b)
            data.appendLE(Double(1.0))                        // error（無重投影資訊，設常數）
            data.appendLE(UInt64(0))                          // 空 track
        }
        try data.write(to: url, options: [.atomic])
    }

    // MARK: - PLY（快速檢視 / nerfstudio 種子點用的副本）

    static func writePLY(_ points: [CloudPoint], to url: URL) throws {
        var data = Data(capacity: points.count * 15 + 512)
        var header = "ply\nformat binary_little_endian 1.0\n"
        header += "comment fable ARKit capture (world: ARKit gravity-aligned, Y-up, meters)\n"
        header += "element vertex \(points.count)\n"
        header += "property float x\nproperty float y\nproperty float z\n"
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        header += "end_header\n"
        data.append(header.data(using: .ascii)!)
        for p in points {
            data.appendLE(p.x)
            data.appendLE(p.y)
            data.appendLE(p.z)
            data.append(p.r)
            data.append(p.g)
            data.append(p.b)
        }
        try data.write(to: url, options: [.atomic])
    }

    // MARK: - meta / 修正後姿態 / zip

    /// 錨點修正後的姿態另存一份 jsonl（sidecar）：
    /// poses.jsonl 保留採集當下的原始 VIO 姿態，本檔為 sparse/0 實際使用的版本，
    /// tools/（arkit2gs、validate）偵測到本檔會優先讀取。
    static func writeRefinedPoses(_ records: [FrameRecord], to url: URL) throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys]
        var data = Data()
        for r in records {
            data.append(try enc.encode(r))
            data.append(0x0A)
        }
        try data.write(to: url, options: [.atomic])
    }

    static func writeMeta(_ meta: SessionMeta, to url: URL) throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try enc.encode(meta).write(to: url, options: [.atomic])
    }

    /// 目錄 → .zip。利用 NSFileCoordinator 的 .forUploading 讀取選項：
    /// 系統會在協調讀取時自動把目錄壓成 zip 暫存檔（AirDrop / Files 同款機制），
    /// 免任何第三方相依。同步阻塞，請在背景 Task 呼叫。
    static func zipDirectory(_ dir: URL, to dest: URL) throws {
        try CaptureMetadata.migrateLegacyFile(in: dir)
        var coordinatorError: NSError?
        var innerError: Error?
        NSFileCoordinator().coordinate(readingItemAt: dir, options: [.forUploading],
                                       error: &coordinatorError) { zipURL in
            do {
                if FileManager.default.fileExists(atPath: dest.path) {
                    try FileManager.default.removeItem(at: dest)
                }
                try FileManager.default.copyItem(at: zipURL, to: dest)
            } catch {
                innerError = error
            }
        }
        if let coordinatorError { throw coordinatorError }
        if let innerError { throw innerError }
    }
}
