// swift-tools-version: 6.2
import PackageDescription

// MetalLab: the engine (loader, models, current kernels, drafter, generation); what the server will use.
// LabBench: what only the lab-dev benches use (.npy reading) and the toy models' configs (also in
//   MetalLabTests, toyMatchesReal).
// LabServe: OpenAI-compatible server (BPE tokenizer, chat template with swift-jinja, reasoning parser and
//   tool calls, HTTP with Network.framework) on top of InferenceEngine.
// LabCLI: what the two CLIs share (arguments, engine options, memory plan).
// lab: to use the engine (serve, run, plan, info). lab-dev: exactness tests and benches (gen, kl, reuse, ttft…).
let package = Package(
    name: "ingot",
    platforms: [.macOS("27.0")],
    products: [
        .executable(name: "lab", targets: ["lab"]),
        .executable(name: "lab-dev", targets: ["lab-dev"]),
        .library(name: "MetalLab", targets: ["MetalLab"]),
    ],
    dependencies: [
        // Chat templates (Jinja) with transformers' semantics; Apache-2.0.
        .package(url: "https://github.com/huggingface/swift-jinja.git", exact: "2.5.2")
    ],
    targets: [
        .target(name: "MetalLab", resources: [.copy("Shaders"), .copy("Calibration")]),
        .target(name: "LabBench", dependencies: ["MetalLab"]),
        .target(name: "LabServe", dependencies: ["MetalLab", .product(name: "Jinja", package: "swift-jinja")]),
        .target(name: "LabCLI", dependencies: ["MetalLab"]),
        .executableTarget(name: "lab", dependencies: ["MetalLab", "LabServe", "LabCLI"]),
        .executableTarget(name: "lab-dev", dependencies: ["MetalLab", "LabBench", "LabServe", "LabCLI"]),
        // -O on the kernel tests only: their CPU references and synthetic weights run
        // ~3x faster (KernelCaseTests 203 -> 75 s); the engine keeps the debug build with -enable-testing.
        .testTarget(
            name: "MetalLabTests", dependencies: ["MetalLab", "LabBench"], swiftSettings: [.unsafeFlags(["-O"])]),
        .testTarget(name: "LabServeTests", dependencies: ["LabServe"], resources: [.copy("Resources")]),
    ]
)
