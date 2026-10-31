import Foundation
import Ingot
import Testing

/// `IngotVersion.current` is SemVer and CHANGELOG.md has its section (Keep a Changelog). Reads the file: no GPU.
/// The date of the heading is checked by the release script, not here.
@Suite struct VersionTests {
    static let changelog = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().appendingPathComponent("CHANGELOG.md")

    @Test func versionIsSemVer() {
        #expect(IngotVersion.current.wholeMatch(of: /\d+\.\d+\.\d+/) != nil, "not SemVer: \(IngotVersion.current)")
    }

    @Test func changelogHasVersionSection() throws {
        let headings = try String(contentsOf: Self.changelog, encoding: .utf8).split(separator: "\n")
            .filter { $0.hasPrefix("## ") }
        #expect(headings.contains("## [Unreleased]"), "CHANGELOG.md has no ## [Unreleased]")
        let own = "## [\(IngotVersion.current)]"
        #expect(
            headings.contains { $0 == own || $0.hasPrefix(own + " - ") },
            "CHANGELOG.md has no \(own) section")
    }
}
