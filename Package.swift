// swift-tools-version: 6.2
import PackageDescription

// Ingot: the engine (loader, models, current kernels, drafter, generation); what the server will use.
// IngotBench: what only the ingot-dev benches use (.npy reading) and the toy models' configs (also in
//   IngotTests, toyMatchesReal).
// IngotServe: OpenAI-compatible server (BPE tokenizer, chat template with swift-jinja, reasoning parser and
//   tool calls, HTTP with Network.framework) on top of InferenceEngine.
// IngotCLI: what the two CLIs share (arguments, engine options, memory plan).
// ingot (target IngotTool: a target named `ingot` would clash with the Ingot module on a case-insensitive
//   file system): to use the engine (serve, run, plan, info). ingot-dev: exactness tests and benches (gen, kl, reuse, ttft…).
let package = Package(
    name: "ingot",
    platforms: [.macOS("27.0")],
    products: [
        .executable(name: "ingot", targets: ["IngotTool"]),
        .executable(name: "ingot-dev", targets: ["ingot-dev"]),
        .library(name: "Ingot", targets: ["Ingot"]),
    ],
    dependencies: [
        // Chat templates (Jinja) with transformers' semantics; Apache-2.0.
        .package(url: "https://github.com/huggingface/swift-jinja.git", exact: "2.5.2")
    ],
    targets: [
        .target(name: "Ingot", resources: [.copy("Shaders"), .copy("Calibration")]),
        .target(name: "IngotBench", dependencies: ["Ingot"]),
        .target(name: "IngotServe", dependencies: ["Ingot", .product(name: "Jinja", package: "swift-jinja")]),
        .target(name: "IngotCLI", dependencies: ["Ingot"]),
        .executableTarget(name: "IngotTool", dependencies: ["Ingot", "IngotServe", "IngotCLI"]),
        .executableTarget(name: "ingot-dev", dependencies: ["Ingot", "IngotBench", "IngotServe", "IngotCLI"]),
        // -O on the kernel tests only: their CPU references and synthetic weights run
        // ~3x faster (KernelCaseTests 203 -> 75 s); the engine keeps the debug build with -enable-testing.
        .testTarget(
            name: "IngotTests", dependencies: ["Ingot", "IngotBench"], swiftSettings: [.unsafeFlags(["-O"])]),
        .testTarget(name: "IngotServeTests", dependencies: ["IngotServe"], resources: [.copy("Resources")]),
    ]
)
