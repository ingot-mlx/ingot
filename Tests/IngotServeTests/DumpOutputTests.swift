import Foundation
import Testing

@testable import IngotServe

/// `--dump-requests` writes what each request generated: the ids ref/live_diff.py compares with the next prompt.
@Suite struct DumpOutputTests {
    @Test func outputRecordRoundTrips() throws {
        let v = ChatServer.dumpOutput(
            promptIds: [1, 2, 3], outputIds: [7, 8, 106], finish: "tool_calls", cached: 2, cacheSource: "live",
            rolledBackFrom: 9, liveMiss: "live diverged at 5 (answer token 2 of 6)")
        let back = try JSONValue.parse(v.compact())
        #expect(back["prompt_ids"]?.array?.compactMap(\.intValue) == [1, 2, 3])
        #expect(back["output_ids"]?.array?.compactMap(\.intValue) == [7, 8, 106])
        #expect(back["finish"]?.string == "tool_calls")
        #expect(back["cached"]?.intValue == 2)
        #expect(back["cache_source"]?.string == "live")
        #expect(back["rolled_back_from"]?.intValue == 9)
        #expect(back["live_miss"]?.string == "live diverged at 5 (answer token 2 of 6)")
    }

    @Test func absentFieldsAreNull() throws {
        let v = ChatServer.dumpOutput(
            promptIds: [1], outputIds: [2], finish: "stop", cached: 0, cacheSource: "none", rolledBackFrom: nil,
            liveMiss: nil)
        let back = try JSONValue.parse(v.compact())
        #expect(back["rolled_back_from"]?.isNull == true)
        #expect(back["live_miss"]?.isNull == true)
    }
}
