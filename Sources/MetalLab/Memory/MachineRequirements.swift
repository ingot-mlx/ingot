import Foundation
import Metal

/// What the engine assumes about the machine: a GPU with matrix accelerators (the NAX kernels use
/// `mpp::tensor_ops::matmul2d`, Metal Performance Primitives) and macOS 27 (`languageVersion = .version4_1`, SDK 27).
/// Public API only: `MTLDevice.supportsFamily(.apple10)` (first GPU family of the M5/A19 generation, the one that
/// brings matrix accelerators in every core; `MTLGPUFamilyApple10` in `MTLDevice.h`) and `.metal4`.
public enum MachineRequirements {
    public static let minimumMajorOS = 27

    /// What is known about the machine, separate from the real Mac so the decision can be tested.
    public struct Capabilities: Sendable, Equatable {
        public var chipName: String
        public var osMajor: Int
        public var osVersion: String
        public var apple10: Bool
        public var metal4: Bool

        public init(chipName: String, osMajor: Int, osVersion: String, apple10: Bool, metal4: Bool) {
            self.chipName = chipName
            self.osMajor = osMajor
            self.osVersion = osVersion
            self.apple10 = apple10
            self.metal4 = metal4
        }

        public static func current(device: MTLDevice) -> Capabilities {
            let v = ProcessInfo.processInfo.operatingSystemVersion
            return Capabilities(
                chipName: device.name, osMajor: v.majorVersion,
                osVersion: "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)",
                apple10: device.supportsFamily(.apple10), metal4: device.supportsFamily(.metal4))
        }
    }

    /// What is missing (empty = all good).
    public static func missing(_ c: Capabilities) -> [String] {
        var m: [String] = []
        if !c.apple10 {
            m.append("GPU with matrix accelerators (Metal Apple10 family, the one in M5 chips or later)")
        }
        if !c.metal4 { m.append("Metal 4") }
        if c.osMajor < minimumMajorOS { m.append("macOS \(minimumMajorOS) or later") }
        return m
    }

    public static var skipRequested: Bool { ProcessInfo.processInfo.environment["METAL_LAB_SKIP_CHECK"] == "1" }

    /// Throws a clear error if the machine does not meet them. `METAL_LAB_SKIP_CHECK=1` skips it (diagnostics only).
    public static func verify(_ c: Capabilities, skip: Bool = skipRequested) throws {
        let m = missing(c)
        if m.isEmpty || skip { return }
        throw SafetensorsError.io(
            "this machine does not meet metal-lab's requirements: detected chip «\(c.chipName)», macOS \(c.osVersion). "
                + "Missing: \(m.joined(separator: "; ")). A Mac with an M5 or later chip (GPU with NAX/tensor ops) is required "
                + "and macOS 27 or later.")
    }

    public static func verify(device: MTLDevice) throws { try verify(Capabilities.current(device: device)) }
}
