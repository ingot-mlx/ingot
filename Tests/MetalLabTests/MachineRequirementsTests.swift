import Testing

@testable import MetalLab

/// Machine requirements decision, with synthetic capabilities (does not depend on the real Mac).
struct MachineRequirementsTests {
    typealias Caps = MachineRequirements.Capabilities
    static let m6 = Caps(chipName: "Apple M6", osMajor: 27, osVersion: "27.0.0", apple10: true, metal4: true)

    @Test func m6Cumple() throws {
        #expect(MachineRequirements.missing(Self.m6).isEmpty)
        try MachineRequirements.verify(Self.m6, skip: false)
    }

    @Test func m4SinAceleradores() {
        var c = Self.m6
        c.chipName = "Apple M4"
        c.apple10 = false
        #expect(MachineRequirements.missing(c).count == 1)
        #expect(throws: (any Error).self) { try MachineRequirements.verify(c, skip: false) }
        do {
            try MachineRequirements.verify(c, skip: false)
        } catch {
            let t = "\(error)"
            #expect(t.contains("Apple M4") && t.contains("Apple10") && t.contains("macOS 27"))
        }
    }

    @Test func macOSAntiguo() {
        var c = Self.m6
        c.osMajor = 26
        c.osVersion = "26.2.0"
        #expect(MachineRequirements.missing(c) == ["macOS 27 or later"])
    }

    @Test func skipCheckEnvironment() throws {
        var c = Self.m6
        c.apple10 = false
        c.metal4 = false
        #expect(MachineRequirements.missing(c).count == 2)
        try MachineRequirements.verify(c, skip: true)
    }
}
