import Foundation
import LabCLI
import MetalLab

/// Distance between two logits vectors against the reference `ref`: max |Δlogit| and NMSE (Σ(x − ref)² / Σ ref²,
/// as in llama.cpp's test-llama-archs). The tests are bit-exact; this only says how big the difference is when one fails.
func logitDistance(_ x: [Float], _ ref: [Float]) -> (maxAbs: Float, nmse: Double) {
    var mx: Float = 0, num = 0.0, den = 0.0
    for (a, b) in zip(x, ref) {
        let d = a - b
        mx = max(mx, abs(d))
        num += Double(d) * Double(d)
        den += Double(b) * Double(b)
    }
    return (mx, den > 0 ? num / den : (num > 0 ? .infinity : 0))
}

/// "max |Δlogit| …, NMSE …" for the test lines when two vectors are not bit-identical.
func logitDistanceNote(_ x: [Float], _ ref: [Float]) -> String {
    let d = logitDistance(x, ref)
    return String(format: "max |Δlogit| %.3g, NMSE %.3g", d.maxAbs, d.nmse)
}

/// Index of the largest element (the last of equal maxima, as `max(by:)`; 0 if empty).
func argmaxIndex<T: Comparable>(_ a: [T]) -> Int { a.indices.max { a[$0] < a[$1] } ?? 0 }

/// log Σ exp(v) in fp64, shifted by the maximum.
func logSumExp(_ v: [Float]) -> Double {
    let mx = Double(v.max()!)
    return mx + log(v.reduce(0.0) { $0 + exp(Double($1) - mx) })
}

/// First position where two id lists differ (nil if one is a prefix of the other).
func firstDifference(_ a: [UInt32], _ b: [UInt32]) -> Int? { zip(a, b).enumerated().first { $1.0 != $1.1 }?.offset }

/// Compares the logits of the steps of a `tail()` (reuse, prefixcache): one line per step; if not bit-identical,
/// top-1, KL(a ‖ b) in nats (to tell numeric noise from a wrong state), max |Δlogit| and NMSE.
func compareLogitSteps(_ a: [[Float]], _ b: [[Float]], _ what: String) -> Bool {
    var ok = true
    for (i, (x, y)) in zip(a, b).enumerated() {
        let d = zip(x, y).map { abs($0 - $1) }.max()!
        let same = x == y
        ok = ok && same
        var extra = ""
        if !same {
            let ax = argmaxIndex(x), ay = argmaxIndex(y)
            let lx = logSumExp(x), ly = logSumExp(y)
            var kl = 0.0
            for j in 0..<x.count {
                let a = Double(x[j]) - lx; if a > -30 { kl += exp(a) * (a - (Double(y[j]) - ly)) }
            }
            extra = String(
                format: "; top-1 %@ (%d / %d), KL %.2e, %@", ax == ay ? "same" : "DIFFERENT", ax, ay, kl,
                logitDistanceNote(x, y))
        }
        print(
            String(
                format: "  %@ step %d: %@ (max |Δ| %.3g%@)", what, i, same ? "bit-identical" : "DIFFERENT", d,
                extra))
    }
    return ok
}

/// Negative control of ci.sh --toy (hidden option `--perturb` of reuse, prefixcache and splitcheck; off by default):
/// alters the recurrent state of the first layer that has one (GDN: +0.01 to head 0 of the state; short convolution:
/// +0.01 to its history) right after restoring a snapshot or an entry, or between two prefill chunks. The test has
/// to report FAILED; if it reports OK, it is not checking what it claims. It does not touch the engine. (rowscheck
/// without a drafter reads `--perturb` itself: it changes the route of its rounds, see `rowsStepsCheck`.)
@MainActor func perturbForTest(_ m: HybridModel, _ what: String) {
    guard args.flags.contains("perturb") else { return }
    func add(_ p: UnsafeMutablePointer<Float>, _ n: Int) { for i in 0..<n { p[i] += 0.01 } }
    for l in m.layers {
        switch l.mixer {
        case .gdn(let g):
            let n = g.dv * g.dk
            add(g.state[g.cur].contents().bindMemory(to: Float.self, capacity: n), n)
            print("  [negative control] GDN state of layer \(g.layer) perturbed (\(what))")
            return
        case .conv(let c):
            let n = c.hist[c.cur].length / 4
            add(c.hist[c.cur].contents().bindMemory(to: Float.self, capacity: n), n)
            print("  [negative control] convolution history of layer \(c.layer) perturbed (\(what))")
            return
        case .attn, .none: continue
        }
    }
    // The latent MLA (GLM, attention only): c's scale of every cached position of the first layer × 1.05.
    for l in m.layers {
        guard case .attn(let a) = l.mixer, a.latentKV else { continue }
        let p = a.vs.contents().bindMemory(to: Float.self, capacity: a.pos)
        for i in 0..<a.pos { p[i] *= 1.05 }
        print("  [negative control] latent KV scales of layer \(a.layer) perturbed (\(what))")
        return
    }
    print("  [negative control] the model has no layers with recurrent state: nothing is perturbed")
}
