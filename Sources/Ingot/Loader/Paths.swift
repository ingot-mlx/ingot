import Foundation

public enum ModelPaths {
    /// Accepts a path to a directory (with a leading `~`) or a Hugging Face id (`org/repo`, looks in the local
    /// cache; does not download).
    public static func resolve(_ s0: String) throws -> URL {
        let s = (s0 as NSString).expandingTildeInPath
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: s, isDirectory: &isDir), isDir.boolValue {
            return URL(fileURLWithPath: s)
        }
        let hub =
            ProcessInfo.processInfo.environment["HF_HUB_CACHE"]
            ?? (NSHomeDirectory() + "/.cache/huggingface/hub")
        let snaps = URL(fileURLWithPath: hub)
            .appendingPathComponent("models--" + s.replacingOccurrences(of: "/", with: "--"))
            .appendingPathComponent("snapshots")
        let refMain = snaps.deletingLastPathComponent().appendingPathComponent("refs/main")
        if let rev = try? String(contentsOf: refMain, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines),
            FileManager.default.fileExists(atPath: snaps.appendingPathComponent(rev).path)
        {
            return snaps.appendingPathComponent(rev)
        }
        guard let first = try? FileManager.default.contentsOfDirectory(atPath: snaps.path).sorted().first else {
            throw SafetensorsError.io("cannot find \(s) in \(hub)")
        }
        return snaps.appendingPathComponent(first)
    }

    /// Short name for reference directories.
    public static func shortName(_ s: String) -> String {
        (s.split(separator: "/").last.map(String.init) ?? s)
    }
}
