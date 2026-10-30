import Foundation
import Metal
import MetalLab

// Shared pieces of the two CLIs: `lab` (for using the engine: serve, info, plan, run) and `lab-dev` (exactness and benchmarks).

/// Command line: `command [positionals…] [--option value | --flag]…`.
package struct Args: Sendable {
    package var cmd = "help"
    /// Loose words after the command (e.g. `lab-dev hw peak`).
    package var positional: [String] = []
    package var opts: [String: String] = [:]
    package var flags: Set<String> = []
    /// `--exp key=value` (repeatable; the last of a repeated key wins): EngineOptions.exp.
    package var exp: [String: String] = [:]

    package init(_ arguments: [String] = Array(CommandLine.arguments.dropFirst())) {
        var it = arguments.makeIterator()
        if let c = it.next() { cmd = c }
        var pending: String?
        while let a = it.next() {
            if a.hasPrefix("--") {
                if let p = pending { flags.insert(p) }
                pending = String(a.dropFirst(2))
            } else if let p = pending {
                if p == "exp" {
                    let kv = a.split(separator: "=", maxSplits: 1).map(String.init)
                    exp[kv[0]] = kv.count > 1 ? kv[1] : ""
                } else {
                    opts[p] = a
                }
                pending = nil
            } else {
                positional.append(a)
            }
        }
        if let p = pending { flags.insert(p) }
    }
    package func int(_ k: String, _ d: Int) -> Int { opts[k].flatMap(Int.init) ?? d }
    package func str(_ k: String, _ d: String) -> String { opts[k] ?? d }
    /// The value of a mandatory option; a usage error if it is missing.
    package func required(_ k: String, _ what: String) throws -> String {
        guard let v = opts[k] else { throw UsageError("--\(k) is required (\(what))") }
        return v
    }
}

/// What `--model` takes, for the error when it is missing.
package let modelHelp = "a model directory or a Hugging Face org/repo in the local cache"

/// The arguments of this process.
package let args = Args()

/// Repo root (tokenized prompts and ref/out references; only the lab-dev benchmarks use it).
package let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent()

package func fmtBytes(_ b: Int) -> String { String(format: "%.1f MB", Double(b) / 1e6) }

/// Command-line usage error.
package struct UsageError: Error, CustomStringConvertible {
    package let description: String
    package init(_ d: String) { description = d }
}

/// Directory, config and a MetalContext with the model's dimensions.
package func loadModelFiles(_ name: String) throws -> (dir: URL, cfg: ModelConfig, ctx: MetalContext) {
    let dir = try ModelPaths.resolve(name)
    let cfg = try ModelConfig(directory: dir)
    // An unknown family errors out here (it used to fall back to Qwen silently).
    return (dir, cfg, try MetalContext(dims: try cfg.requiredSpec.shape))
}

/// The model's drafter, or nil for none (DFlash2Drafter.resolve: --drafter none or a name, the toy's `drafter/` if
/// `toyPack`, the pack's `drafter/`, or the family's if it fits). The line saying the family's does not fit is printed
/// once per process.
package func resolvedDrafterDir(_ modelDir: URL, _ cfg: ModelConfig, toyPack: Bool) throws -> URL? {
    try DFlash2Drafter.resolve(modelDir: modelDir, config: cfg, option: args.opts["drafter"], toyPack: toyPack) {
        if drafterNotices.insert($0).inserted { print($0) }
    }
}
nonisolated(unsafe) private var drafterNotices: Set<String> = []

/// Runs the command and exits with 1 (and the error on stderr) if it fails.
/// With `--pipelines FILE`, at the end appends to the file the kernels the command requested (one per line, no repeats).
package func runCommand(_ body: () throws -> Void) {
    setvbuf(stdout, nil, _IOLBF, 0)  // line-buffered output even when piped
    do {
        try body()
        if let path = args.opts["pipelines"] { try recordPipelines(path) }
    } catch {
        FileHandle.standardError.write(Data("error: \(error)\n".utf8))
        exit(1)
    }
}

/// Merges the kernels requested in this process with those the file already has.
package func recordPipelines(_ path: String) throws {
    let old = (try? String(contentsOfFile: path, encoding: .utf8))?.split(separator: "\n").map(String.init) ?? []
    let all = Set(old).union(MetalContext.usedPipelines).sorted()
    try (all.joined(separator: "\n") + "\n").write(toFile: path, atomically: true, encoding: .utf8)
}
