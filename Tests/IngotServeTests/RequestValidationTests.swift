import Foundation
import Testing

@testable import IngotServe

/// Sampling fields the engine does not implement: 400 when not neutral, accepted when neutral or absent.
struct RequestValidationTests {
    static func rejected(_ json: String) -> Bool {
        do {
            try ChatServer.rejectUnsupported(try JSONValue.parse(json))
            return false
        } catch is APIError {
            return true
        } catch {
            Issue.record("unexpected error \(error)")
            return false
        }
    }

    @Test func neutralValuesPass() {
        for body in [
            #"{}"#, #"{"n": 1}"#, #"{"logit_bias": {}}"#, #"{"logit_bias": null}"#, #"{"logprobs": false}"#,
            #"{"top_logprobs": 0}"#, #"{"xtc_probability": 0}"#, #"{"xtc_probability": 0.0, "xtc_threshold": 0.1}"#,
            #"{"repetition_context_size": null}"#,
        ] {
            #expect(!Self.rejected(body), "\(body)")
        }
    }

    @Test func nonNeutralValuesAre400() {
        for body in [
            #"{"n": 2}"#, #"{"logit_bias": {"42": -100}}"#, #"{"logprobs": true}"#, #"{"top_logprobs": 5}"#,
            #"{"xtc_probability": 0.5}"#, #"{"repetition_context_size": 20}"#, #"{"presence_context_size": 20}"#,
            #"{"frequency_context_size": 64}"#,
        ] {
            #expect(Self.rejected(body), "\(body)")
        }
    }
}
