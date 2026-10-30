import Foundation
import Metal

/// What the memory planner needs to know about the Mac (all public API). Another memory size can be simulated with
/// `simulating(memoryGB:)` (`--mem-limit-gb N`) to test the 16, 24 or 48 GB policies on this Mac.
public struct MachineInfo: Sendable, Equatable {
    /// `ProcessInfo.physicalMemory` (bytes).
    public var physicalMemory: Int
    /// `MTLDevice.recommendedMaxWorkingSetSize`: what macOS lets the GPU use without performance dropping.
    public var recommendedWorkingSet: Int
    /// `MTLDevice.maxBufferLength`: no tensor or buffer can exceed this.
    public var maxBufferLength: Int
    /// Memory available now (free + inactive + purgeable, `host_statistics64`); only for warnings. nil = unknown
    /// (or simulated machine).
    public var availableMemory: Int?
    public var name: String
    /// Description of the simulation (nil = the real memory).
    public var simulated: String?

    public init(
        physicalMemory: Int, recommendedWorkingSet: Int, maxBufferLength: Int, availableMemory: Int? = nil,
        name: String = "synthetic", simulated: String? = nil
    ) {
        self.physicalMemory = physicalMemory
        self.recommendedWorkingSet = recommendedWorkingSet
        self.maxBufferLength = maxBufferLength
        self.availableMemory = availableMemory
        self.name = name
        self.simulated = simulated
    }

    /// The real machine (with the default Metal device, or the one passed in).
    public static func current(device: MTLDevice? = nil) -> MachineInfo {
        let phys = Int(ProcessInfo.processInfo.physicalMemory)
        guard let d = device ?? MTLCreateSystemDefaultDevice() else {
            return MachineInfo(
                physicalMemory: phys, recommendedWorkingSet: phys * 2 / 3, maxBufferLength: phys / 2,
                availableMemory: availableNow(), name: "no GPU")
        }
        return MachineInfo(
            physicalMemory: phys, recommendedWorkingSet: Int(d.recommendedMaxWorkingSetSize),
            maxBufferLength: d.maxBufferLength, availableMemory: availableNow(), name: d.name)
    }

    /// A Mac of `memoryGB` GB (as they are sold: GiB; 32 reproduces this M6). GPU memory and the maximum buffer
    /// scale with this machine's ratio (here, rws = 78% of physical), unless `rwsGB` is given (in decimal GB,
    /// as `ingot plan` prints them). It is an estimate: on other Macs the percentage may differ.
    public func simulating(memoryGB: Double, rwsGB: Double? = nil) -> MachineInfo {
        let phys = Int(memoryGB * 1_073_741_824)
        let ratio = Double(recommendedWorkingSet) / Double(physicalMemory)
        let rws = rwsGB.map { Int($0 * 1e9) } ?? Int(Double(phys) * ratio)
        let maxBuf = Int(Double(phys) * Double(maxBufferLength) / Double(physicalMemory))
        let how =
            rwsGB != nil
            ? String(format: "rws %.2f GB given", rwsGB!)
            : String(format: "rws = %.0f%% of physical, as on this Mac", ratio * 100)
        return MachineInfo(
            physicalMemory: phys, recommendedWorkingSet: rws, maxBufferLength: maxBuf, availableMemory: nil,
            name: name, simulated: String(format: "simulating a %g GB Mac (%@)", memoryGB, how))
    }

    /// Free + inactive + purgeable, in bytes (nil if `host_statistics64` fails).
    public static func availableNow() -> Int? {
        var st = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &st) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return nil }
        let page = Int(getpagesize())
        return (Int(st.free_count) + Int(st.inactive_count) + Int(st.purgeable_count)) * page
    }
}
