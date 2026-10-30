import Accelerate
import Foundation
import Metal

@testable import Ingot

// Harness of the per-kernel tests (KernelCaseTests): small synthetic tensors, matrices quantized in the engine's
// formats and CPU references. The "one 1 per row" matrices (oneHot4) make a quantized projection an exact permutation
// of columns: a whole layer (launchers included) is then compared with the CPU without having to reproduce the GEMM,
// and the remaining error is only that of the part under test.

/// Tolerance of a check.
enum Tol {
    /// Same bits (row invariance, indices).
    case bitExact
    /// max |gpu − ref| / max |ref|.
    case maxRel(Float)
    /// ‖gpu − ref‖ / ‖ref‖: lossy kernels (bounded error).
    case rms(Float)
    /// At most n half ulps apart (values that are halves on both sides: elementwise kernels with half output).
    case halfUlp(Int)
}

struct Check {
    let label: String
    let gpu: [Float]
    let ref: [Float]
    let tol: Tol

    /// (ok, measured error, text for the report).
    func evaluate() -> (ok: Bool, err: Float, text: String) {
        guard gpu.count == ref.count, !ref.isEmpty else {
            return (false, .infinity, "\(label): \(gpu.count) values against \(ref.count)")
        }
        switch tol {
        case .bitExact:
            let bad = zip(gpu, ref).filter { $0.bitPattern != $1.bitPattern }.count
            return (bad == 0, Float(bad), "\(label): \(bad) of \(ref.count) differ (bit by bit)")
        case .maxRel(let t):
            var mx: Float = 0, mr: Float = 0
            for (a, b) in zip(gpu, ref) {
                mx = max(mx, abs(a - b))
                mr = max(mr, abs(b))
                if a.isNaN { mx = .infinity }
            }
            let e = mx / max(mr, 1e-30)
            return (e <= t, e, "\(label): max relative error \(e) (tolerance \(t))")
        case .rms(let t):
            var se: Double = 0, sr: Double = 0
            for (a, b) in zip(gpu, ref) {
                se += Double(a - b) * Double(a - b)
                sr += Double(b) * Double(b)
            }
            let e = Float((se / max(sr, 1e-300)).squareRoot())
            return (e <= t, e, "\(label): relative RMS error \(e) (bound \(t))")
        case .halfUlp(let n):
            func ord(_ v: Float) -> Int {
                let b = Int(Float16(v).bitPattern)
                return b & 0x8000 != 0 ? -(b & 0x7FFF) : b
            }
            let d = zip(gpu, ref).map { $0.isNaN ? Int.max : abs(ord($0) - ord($1)) }.max() ?? 0
            return (d <= n, Float(d), "\(label): \(d) half ulps max (tolerance \(n))")
        }
    }
}

// MARK: - Synthetic data

enum Syn {
    /// Normal(0, sd) with Box-Muller.
    static func normal(_ g: inout SplitMix64, _ n: Int, sd: Float = 1) -> [Float] {
        (0..<n).map { _ in
            let u1 = max(g.uniform(), 1e-7), u2 = g.uniform()
            return sd * (-2 * log(u1)).squareRoot() * cos(2 * .pi * u2)
        }
    }

    static func roundHalf(_ v: [Float]) -> [Float] { v.map { Float(Float16($0)) } }
    /// Slab e of `n` values: expert e of stacked weights.
    static func slab(_ a: [Float], _ e: Int, _ n: Int) -> [Float] { Array(a[(e * n)..<((e + 1) * n)]) }
    static func roundBF16(_ v: [Float]) -> [Float] { v.map { bf16ToFloat(floatToBF16($0)) } }

    static func halfBuffer(_ ctx: MetalContext, _ v: [Float]) -> MTLBuffer { ctx.buffer(v.map { Float16($0) }) }
    static func bf16Buffer(_ ctx: MetalContext, _ v: [Float]) -> MTLBuffer { ctx.buffer(v.map { floatToBF16($0) }) }

    static func floats(_ b: MTLBuffer, _ n: Int, offset: Int = 0) -> [Float] {
        let p = (b.contents() + offset).bindMemory(to: Float.self, capacity: n)
        return Array(UnsafeBufferPointer(start: p, count: n))
    }

    static func uints(_ b: MTLBuffer, _ n: Int) -> [UInt32] {
        Array(UnsafeBufferPointer(start: b.contents().bindMemory(to: UInt32.self, capacity: n), count: n))
    }

    /// Random MLX 4-bit matrix (groups of `group`, 64 or 32; low nibble first, bf16 scale and bias) and its float
    /// value.
    static func q4(_ ctx: MetalContext, N: Int, K: Int, _ g: inout SplitMix64, group G: Int = 64) -> (
        m: DenseQ, w: [Float]
    ) {
        var words = [UInt32](repeating: 0, count: N * K / 8), w = [Float](repeating: 0, count: N * K)
        var s = [UInt16](repeating: 0, count: N * K / G), b = s
        for n in 0..<N {
            for gi in 0..<(K / G) {
                let sc = floatToBF16(0.01 + 0.02 * g.uniform()), bi = floatToBF16(-8 * bf16ToFloat(sc) * g.uniform())
                s[n * K / G + gi] = sc
                b[n * K / G + gi] = bi
                for j in 0..<G {
                    let k = gi * G + j, q = UInt32(g.next() & 15)
                    words[(n * K + k) / 8] |= q << UInt32(4 * (k % 8))
                    w[n * K + k] = bf16ToFloat(sc) * Float(q) + bf16ToFloat(bi)
                }
            }
        }
        return (DenseQ(w: ctx.buffer(words), s: ctx.buffer(s), b: ctx.buffer(b), N: N, K: K, bits: 4, gs: G), w)
    }

    /// 4-bit matrix with a single 1 per row (row n: column cols[n]): y = x·Wᵀ is an exact permutation of x.
    static func oneHot4(_ ctx: MetalContext, N: Int, K: Int, cols: [Int]) -> DenseQ {
        var words = [UInt32](repeating: 0, count: N * K / 8)
        for n in 0..<N where cols[n] >= 0 { words[(n * K + cols[n]) / 8] |= 1 << UInt32(4 * (cols[n] % 8)) }
        let s = [UInt16](repeating: floatToBF16(1), count: N * K / 64), b = [UInt16](repeating: 0, count: N * K / 64)
        return DenseQ(w: ctx.buffer(words), s: ctx.buffer(s), b: ctx.buffer(b), N: N, K: K, bits: 4)
    }

    /// Bonsai 2 ternary (q ∈ {0, 1, 2} in 2 bits, w = (q − 1)·s, fp16 scale per group of 128) with the signs of the
    /// input's Hadamard rotation. Also returns W in float and the signs.
    static func ternary(_ ctx: MetalContext, N: Int, K: Int, _ g: inout SplitMix64) -> (
        m: DenseQ, w: [Float], signs: [Float]
    ) {
        var words = [UInt32](repeating: 0, count: N * K / 16), w = [Float](repeating: 0, count: N * K)
        var s = [Float16](repeating: 0, count: N * K / 128)
        for n in 0..<N {
            for gi in 0..<(K / 128) {
                let sc = Float16(0.01 + 0.03 * g.uniform())
                s[n * K / 128 + gi] = sc
                for j in 0..<128 {
                    let k = gi * 128 + j, q = UInt32(g.next() % 3)
                    words[(n * K + k) / 16] |= q << UInt32(2 * (k % 16))
                    w[n * K + k] = (Float(q) - 1) * Float(sc)
                }
            }
        }
        let signs = (0..<K).map { _ in g.next() & 1 == 0 ? Float(1) : -1 }
        let sb = ctx.buffer(s)
        var m = DenseQ(w: ctx.buffer(words), s: sb, b: sb, N: N, K: K, bits: 2)
        m.had = ctx.buffer(signs)
        return (m, w, signs)
    }

    /// MLX 6-bit (4 values in 3 bytes, little endian; groups of 64 with bf16 scale and bias).
    static func q6(_ ctx: MetalContext, N: Int, K: Int, _ g: inout SplitMix64) -> (m: DenseQ, w: [Float]) {
        let rb = K * 3 / 4
        var bytes = [UInt8](repeating: 0, count: N * rb), w = [Float](repeating: 0, count: N * K)
        var s = [UInt16](repeating: 0, count: N * K / 64), b = s
        for n in 0..<N {
            for gi in 0..<(K / 64) {
                let sc = floatToBF16(0.002 + 0.004 * g.uniform()), bi = floatToBF16(-32 * bf16ToFloat(sc) * g.uniform())
                s[n * K / 64 + gi] = sc
                b[n * K / 64 + gi] = bi
                for j in 0..<64 {
                    let k = gi * 64 + j, q = Int(g.next() & 63)
                    for bit in 0..<6 where (q >> bit) & 1 == 1 {
                        let p = 6 * k + bit
                        bytes[n * rb + p / 8] |= UInt8(1 << (p % 8))
                    }
                    w[n * K + k] = bf16ToFloat(sc) * Float(q) + bf16ToFloat(bi)
                }
            }
        }
        return (DenseQ(w: ctx.buffer(bytes), s: ctx.buffer(s), b: ctx.buffer(b), N: N, K: K, bits: 6), w)
    }
}

// MARK: - CPU references

enum CPU {
    /// GELU with the tanh approximation (mlx-lm `nn.gelu_approx`; checked against ref/act_ref.py), computed in double
    /// so that the fp32 reference has no cancellation of 1 + tanh in the negative tail.
    static func geluTanh(_ x: Float) -> Float {
        let d = Double(x)
        return Float(0.5 * d * (1 + tanh(0.797_884_560_802_865_4 * (d + 0.044_715 * d * d * d))))
    }
    /// gpt-oss's swiglu_oai (mlx-lm activations.swiglu_oai; checked against ref/act_ref.py): `(clip(u, ±limit) + 1) ·
    /// g·σ(1.702·g)` with `g = min(gate, limit)`.
    static func swigluOAI(_ gate: Float, _ up: Float, limit: Float = 7) -> Float {
        let g = min(gate, limit), u = min(max(up, -limit), limit)
        return (u + 1) * (g / (1 + exp(-1.702 * g)))
    }
    /// Y[M, N] = X[M, K] · W[N, K]ᵀ (Accelerate, in float).
    static func matmulT(_ x: [Float], _ w: [Float], M: Int, N: Int, K: Int) -> [Float] {
        var wt = [Float](repeating: 0, count: K * N), y = [Float](repeating: 0, count: M * N)
        vDSP_mtrans(w, 1, &wt, 1, vDSP_Length(K), vDSP_Length(N))
        vDSP_mmul(x, 1, wt, 1, &y, 1, vDSP_Length(M), vDSP_Length(N), vDSP_Length(K))
        return y
    }

    /// Unnormalized Sylvester Walsh-Hadamard, in place.
    static func fwht(_ v: inout [Float]) {
        var h = 1
        while h < v.count {
            for i in stride(from: 0, to: v.count, by: 2 * h) {
                for j in i..<(i + h) {
                    let a = v[j], b = v[j + h]
                    v[j] = a + b
                    v[j + h] = a - b
                }
            }
            h *= 2
        }
    }

    /// Bonsai input rotation: x ← H₁₀₂₄(x ⊙ signs) / 32 per block of 1024, rounded to half.
    static func rotate(_ x: [Float], rows: Int, K: Int, signs: [Float]) -> [Float] {
        var y = x
        for r in 0..<rows {
            for b0 in stride(from: 0, to: K, by: 1024) {
                var v = (0..<1024).map { x[r * K + b0 + $0] * signs[b0 + $0] }
                fwht(&v)
                for i in 0..<1024 { y[r * K + b0 + i] = Float(Float16(v[i] / 32)) }
            }
        }
        return y
    }

    /// Emulated rotated W8A8 of the prefill (xa_dense_i8.metal): unnormalized Hadamard of n = 32·2^lb per contiguous
    /// block of x and of W, int8 per row (scale max/127, round half to even) and Y = (Xq·Wqᵀ)·sx·sw / n.
    static func w8a8(_ x: [Float], _ w: [Float], M: Int, N: Int, K: Int, lb: Int) -> [Float] {
        let n = 32 << lb
        func quantRows(_ a: [Float], _ rows: Int) -> ([Float], [Float]) {
            var q = a, sc = [Float](repeating: 0, count: rows)
            for r in 0..<rows {
                for b0 in stride(from: 0, to: K, by: n) {
                    var v = Array(a[(r * K + b0)..<(r * K + b0 + n)])
                    fwht(&v)
                    q.replaceSubrange((r * K + b0)..<(r * K + b0 + n), with: v)
                }
                let (qq, s) = quantI8(q[(r * K)..<(r * K + K)])
                q.replaceSubrange((r * K)..<(r * K + K), with: qq)
                sc[r] = s
            }
            return (q, sc)
        }
        let (xq, sx) = quantRows(x, M), (wq, sw) = quantRows(w, N)
        var y = matmulT(xq, wq, M: M, N: N, K: K)
        for m in 0..<M { for j in 0..<N { y[m * N + j] *= sx[m] * sw[j] / Float(n) } }
        return y
    }

    /// Symmetric int8 quantization of a vector (like vattn_prework_i8): values and scale.
    static func quantI8(_ x: ArraySlice<Float>) -> ([Float], Float) {
        let amax = x.map { abs($0) }.max() ?? 0
        let sc = amax > 0 ? amax / 127 : 1
        return (x.map { min(127, max(-127, ($0 / sc).rounded(.toNearestOrEven))) }, sc)
    }
}

// MARK: - Attention with synthetic weights

/// AttentionLayer with permutation projections (exact q|k|v and o_proj), random norms and a random int8 KV cache of
/// `P` positions, plus its CPU mirror for the reference. `nope`: layer without RoPE; `unitNorm`: QK-norm without
/// weight (buffer of ones). Without QK-norm: the shape with `attnQKN = false` (LAB_AT_QKN = 0).
final class AttnFixture {
    let ctx: MetalContext
    let a: AttentionLayer
    let dims: ModelDims
    let hidden: Int, hd: Int, qH: Int, kvH: Int, G: Int, qs: Int, projN: Int, outDim: Int
    let colIn: [Int], colOut: [Int]
    let qnw: [Float], knw: [Float]
    let nope: Bool
    /// Attention score scale (AttentionLayer.scale; 1/√hd by default).
    let scale: Float
    /// Bias of q|k|v (fp32 [projN]) added to the projections (empty = none).
    let bias: [Float]
    /// Sliding window of the layer (0 = full attention).
    let window: Int
    /// Learned sinks per query head (with LAB_AT_SINKS), empty without them.
    let sinks: [Float]
    /// RoPE table of the layer (with LAB_AT_ROPETAB): inv_freq per pair and mscale last; nil without it.
    let ropeTab: [Float]?
    let eps: Float = AttnFixture.eps, theta: Float = AttnFixture.theta
    static let eps: Float = 1e-6, theta: Float = 1e6
    /// Cache mirror: k8[h][t·hd + d], v8[h][t·hd + d] (int8 values as float) and per-position scales.
    /// MLA's latent KV (hd 576): ks is k_pe's scale and vs c's, and v8 is K's latent part (dims 0…63 zero).
    var k8: [[Float]], v8: [[Float]], ks: [[Float]], vs: [[Float]]
    var latent: Bool { hd == 576 }

    init(
        _ ctx: MetalContext, positions P: Int, _ g: inout SplitMix64, nope: Bool = false, unitNorm: Bool = false,
        maxCtx: Int = 2048, scale: Float? = nil, qkvBias: Bool = false, window: Int = 0, ringRows: Int = 2048,
        ropeTable: [Float]? = nil
    ) throws {
        let d = ctx.dims
        // With LAB_AT_ROPETAB every layer passes a table (the default RoPE's if the case gives none).
        let tab =
            d.attnRopeTab
            ? ropeTable ?? RopeTable.make(theta: Double(Self.theta), rotaryDims: d.attnRot, scaling: nil) : nil
        // Sinks around the largest scores (std ~1.5 with the normalized q and k), so they take a real share.
        let sinks = d.attnSinks ? Syn.normal(&g, d.attnQH).map { 2 + $0 } : []
        let hidden = d.hidden, hd = d.attnHD, qH = d.attnQH, kvH = d.attnKVH
        let qs = d.attnGate ? 2 * hd : hd, projN = qH * qs + 2 * kvH * hd, outDim = qH * hd
        // MLA's latent head (576): no V head (V is K's latent part): its projection columns are left empty (-1 →
        // zero row).
        let vZero = { (c: Int) in hd == 576 && c >= qH * qs + kvH * hd }
        let colIn = (0..<projN).map { c in
            let col = Int(g.next() % UInt64(hidden))
            return vZero(c) ? -1 : col
        }
        let colOut = (0..<hidden).map { _ in Int(g.next() % UInt64(outDim)) }
        let nq = Syn.normal(&g, hd, sd: 0.3).map { 1 + $0 }, nk = Syn.normal(&g, hd, sd: 0.3).map { 1 + $0 }
        let ones = [Float](repeating: 1, count: hd)
        let qnw = unitNorm ? ones : d.normF32 ? nq : Syn.roundBF16(nq)
        let knw = unitNorm ? ones : d.normF32 ? nk : Syn.roundBF16(nk)
        let nb = { (v: [Float]) in
            unitNorm ? AttentionLayer.unitNorm(ctx) : d.normF32 ? ctx.buffer(v) : Syn.bf16Buffer(ctx, v)
        }
        let a = try AttentionLayer(
            ctx: ctx, layer: 0, maxCtx: maxCtx,
            proj: Syn.oneHot4(ctx, N: projN, K: hidden, cols: colIn),
            oProj: Syn.oneHot4(ctx, N: hidden, K: outDim, cols: colOut),
            qNorm: nb(qnw), kNorm: nb(knw), eps: Self.eps, theta: Self.theta, nope: nope, window: window,
            sinks: sinks.isEmpty ? nil : ctx.buffer(sinks), ropeTable: tab.map { ctx.buffer($0) },
            ringRows: ringRows)
        a.fixedChunk = HybridModel.decodeChunk(promptTokens: 0, latent: hd == 576, maxCtx: maxCtx)
        a.pos = P
        if let scale { a.scale = scale }
        self.scale = a.scale
        let bias = qkvBias ? Syn.normal(&g, projN, sd: 0.5) : []
        if qkvBias { a.qkvBias = ctx.buffer(bias) }
        self.bias = bias
        // Previous cache: random int8 and scales per position and head (with a ring KV, position t in slot t % ring:
        // only the last `ring` positions survive, which covers every window the checks use).
        let ld = a.ldkv
        func slot(_ t: Int) -> Int { a.ring > 0 ? t % a.ring : t }
        let kc = a.kc.contents().bindMemory(to: Int8.self, capacity: kvH * ld * hd)
        let lat = hd == 576
        let vt = a.vt.contents().bindMemory(to: Int8.self, capacity: lat ? 0 : kvH * ld * hd)
        let ksp = a.ks.contents().bindMemory(to: Float.self, capacity: kvH * ld)
        let vsp = a.vs.contents().bindMemory(to: Float.self, capacity: kvH * ld)
        var k8: [[Float]] = [], v8: [[Float]] = [], ks: [[Float]] = [], vs: [[Float]] = []
        for h in 0..<kvH {
            var kk = [Float](), vv = [Float](), sk = [Float](), sv = [Float]()
            for t in 0..<P {
                // Scales of ~1/73: K and V with RMS ~1, like the new (normalized) keys.
                let sck = 0.011 + 0.005 * g.uniform(), scv = 0.011 + 0.005 * g.uniform()
                ksp[h * ld + slot(t)] = sck; vsp[h * ld + slot(t)] = scv
                sk.append(sck); sv.append(scv)
                for i in 0..<hd {
                    let qk = Int8(Int(g.next() % 255) - 127), qr = Int8(Int(g.next() % 255) - 127)
                    let qv = lat ? (i < 64 ? 0 : qk) : qr
                    kc[(h * ld + slot(t)) * hd + i] = qk
                    if !lat { vt[(h * hd + i) * ld + slot(t)] = qv }
                    kk.append(Float(qk)); vv.append(Float(qv))
                }
            }
            k8.append(kk); v8.append(vv); ks.append(sk); vs.append(sv)
        }
        self.ctx = ctx; self.a = a; dims = d
        self.hidden = hidden; self.hd = hd; self.qH = qH; self.kvH = kvH; G = qH / kvH; self.qs = qs
        self.projN = projN; self.outDim = outDim; self.colIn = colIn; self.colOut = colOut
        self.qnw = qnw; self.knw = knw; self.nope = nope; self.window = window
        self.sinks = sinks
        ropeTab = tab
        self.k8 = k8; self.v8 = v8; self.ks = ks; self.vs = vs
    }

    /// Projection row of row t of x (half [rows, hidden]).
    func projRow(_ x: [Float], _ t: Int) -> [Float] {
        let r = colIn.map { $0 < 0 ? 0 : x[t * hidden + $0] }
        return bias.isEmpty ? r : zip(r, bias).map { $0 + $1 }
    }

    /// RMSNorm with weight (if the shape has QK-norm) and RoPE (first attnRot dimensions, non-interleaved halves;
    /// nothing if the layer has no RoPE) of one head.
    func normRope(_ v: ArraySlice<Float>, _ w: [Float], pos: Int) -> [Float] {
        var x = Array(v)
        if dims.attnQKN {
            let ss = v.reduce(0) { $0 + $1 * $1 }
            let r = 1 / (ss / Float(hd) + eps).squareRoot()
            x = v.enumerated().map { $1 * r * w[$0] }
        }
        if nope { return x }
        let x0 = x, rot = dims.attnRot, rh = rot / 2
        for d in 0..<rot {
            let i = d % rh
            // With a table: angle = pos · inv_freq[i], and the rotated dimensions times mscale (mlx-lm scales x
            // before rotating; the rotation is linear).
            let ang = Float(pos) * (ropeTab?[i] ?? pow(theta, -Float(i) / Float(rh)))
            x[d] = d < rh ? x0[d] * cos(ang) - x0[d + rh] * sin(ang) : x0[d] * cos(ang) + x0[d - rh] * sin(ang)
            if let t = ropeTab { x[d] *= t[rh] }
        }
        return x
    }

    /// Weightless RMSNorm of a v head when the shape has it (Gemma 4's v_norm, LAB_AT_VN); as is otherwise.
    func vNorm(_ v: ArraySlice<Float>) -> [Float] {
        guard dims.attnVN else { return Array(v) }
        let r = 1 / (v.reduce(0) { $0 + $1 * $1 } / Float(hd) + eps).squareRoot()
        return v.map { $0 * r }
    }

    /// Orthonormal Walsh-Hadamard of one head (H/sqrt(hd), in fp32), as vattn_prework_i8 applies it to q and k after
    /// the RoPE: the cache holds H k and the queries leave as H q.
    /// MLA's latent head (576): one transform per block, H₆₄ on [k_pe] and H₅₁₂ on [c] (at_fwht in verify_attn_nax.metal).
    func hadamard(_ v: [Float]) -> [Float] {
        if hd == 576 {
            var pe = Array(v[0..<64]), c = Array(v[64...])
            CPU.fwht(&pe)
            CPU.fwht(&c)
            return pe.map { $0 / 8 } + c.map { $0 / Float(512).squareRoot() }
        }
        var r = v
        CPU.fwht(&r)
        let s = 1 / Float(hd).squareRoot()
        return r.map { $0 * s }
    }

    /// Output of the layer (float [rows, hidden], without residual) for `rows` rows at positions P…P+rows−1, with
    /// causal attention over the previous cache and the earlier rows of the block. Adds its K/V to the cache mirror.
    /// `qk8`: Q·Kᵀ with q in int8 per head and row (scale max/127), like q_quant_i8.
    func reference(_ x: [Float], rows: Int, P: Int, qk8: Bool = false) -> [Float] {
        var qrows: [[Float]] = []
        let pr = (0..<rows).map { projRow(x, $0) }
        for t in 0..<rows {
            var qa = [Float]()
            for h in 0..<qH {
                var qv = Syn.roundHalf(hadamard(normRope(pr[t][(h * qs)..<(h * qs + hd)], qnw, pos: P + t)))
                if qk8 {
                    let (q8, sc) = CPU.quantI8(qv[...])
                    qv = q8.map { $0 * sc }
                }
                qa += qv
            }
            qrows.append(qa)
            for h in 0..<kvH {
                let k0 = qH * qs + h * hd, v0 = qH * qs + kvH * hd + h * hd
                var (kq, sk) = CPU.quantI8(hadamard(normRope(pr[t][k0..<(k0 + hd)], knw, pos: P + t))[...])
                var (vq, sv): ([Float], Float)
                if latent {
                    // Two scales: k_pe (dims 0…63) and c (64…575); V = K's latent part.
                    let kr = hadamard(normRope(pr[t][k0..<(k0 + hd)], knw, pos: P + t))
                    let (pe, spe) = CPU.quantI8(kr[0..<64]), (c, sc) = CPU.quantI8(kr[64...])
                    (kq, sk) = (pe + c, spe)
                    (vq, sv) = ([Float](repeating: 0, count: 64) + c, sc)
                } else {
                    (vq, sv) = CPU.quantI8(vNorm(pr[t][v0..<(v0 + hd)])[...])
                }
                if k8[h].count > (P + t) * hd {
                    k8[h].replaceSubrange(((P + t) * hd)..<((P + t + 1) * hd), with: kq)
                    v8[h].replaceSubrange(((P + t) * hd)..<((P + t + 1) * hd), with: vq)
                    ks[h][P + t] = sk; vs[h][P + t] = sv
                } else {
                    k8[h] += kq; v8[h] += vq; ks[h].append(sk); vs[h].append(sv)
                }
            }
        }
        var attn = [Float](repeating: 0, count: rows * outDim)
        for t in 0..<rows {
            let n = P + t + 1
            for h in 0..<qH {
                let kvh = h / G
                let q = Array(qrows[t][(h * hd)..<((h + 1) * hd)])
                var s = [Float](repeating: 0, count: n)
                var mx = -Float.infinity
                // Sliding window: the row at position n − 1 sees keys j with n − 1 − j < window.
                let lo = window > 0 ? max(0, n - window) : 0
                if latent {
                    // S = scale·(ks·(q_pe · k_pe) + vs·(q_c · k_c)), V = K's latent part (in v8).
                    let qpe = Array(q[0..<64]) + [Float](repeating: 0, count: hd - 64)
                    var spe = [Float](repeating: 0, count: n)
                    vDSP_mmul(k8[kvh], 1, qpe, 1, &spe, 1, vDSP_Length(n), 1, vDSP_Length(hd))
                    let qc = [Float](repeating: 0, count: 64) + Array(q[64...])
                    vDSP_mmul(k8[kvh], 1, qc, 1, &s, 1, vDSP_Length(n), 1, vDSP_Length(hd))
                    for j in 0..<n { s[j] = scale * (spe[j] * ks[kvh][j] + s[j] * vs[kvh][j]); mx = max(mx, s[j]) }
                } else {
                    vDSP_mmul(k8[kvh], 1, q, 1, &s, 1, vDSP_Length(n), 1, vDSP_Length(hd))
                    for j in 0..<n { s[j] = j < lo ? -.infinity : s[j] * scale * ks[kvh][j]; mx = max(mx, s[j]) }
                }
                if !sinks.isEmpty { mx = max(mx, sinks[h]) }
                // A sink joins the denominator and carries no value.
                var l: Float = sinks.isEmpty ? 0 : exp(sinks[h] - mx)
                for j in 0..<n { s[j] = exp(s[j] - mx); l += s[j] }
                for j in 0..<n { s[j] *= vs[kvh][j] / l }
                var o = [Float](repeating: 0, count: hd)
                // o[d] = Σ_j p_j·vs_j·v8[j, d]: v8 is [n, hd] → oᵀ = pᵀ · V.
                vDSP_mmul(s, 1, v8[kvh], 1, &o, 1, 1, vDSP_Length(hd), vDSP_Length(n))
                if latent {
                    // The cache holds H₅₁₂ c: rotate the output back.
                    var c = Array(o[64...])
                    CPU.fwht(&c)
                    o = [Float](repeating: 0, count: 64) + c.map { $0 / Float(512).squareRoot() }
                }
                for d in 0..<hd {
                    let gate: Float = dims.attnGate ? 1 / (1 + exp(-pr[t][h * qs + hd + d])) : 1
                    attn[t * outDim + h * hd + d] = Float(Float16(o[d] * gate))
                }
            }
        }
        var out = [Float](repeating: 0, count: rows * hidden)
        for t in 0..<rows { for n in 0..<hidden { out[t * hidden + n] = attn[t * outDim + colOut[n]] } }
        return out
    }

    /// Round (verification of `rows` rows at P, with the round state on the GPU like the model).
    func round(_ x: [Float], rows: Int, P: Int) throws -> [Float] {
        let xb = Syn.halfBuffer(ctx, x), out = ctx.buffer(length: rows * hidden * 4)
        let st = ctx.buffer([UInt32(P), 0, 0, 0, 0, 0, 0, 0])
        a.pos = P
        try ctx.run { e in try a.encode(e, x: xb, out: out, rows: rows, accumulate: false, state: st) }
        return Syn.floats(out, rows * hidden)
    }

    /// Prefill chunk of `rows` rows (padded to M = 64·⌈rows/64⌉) at P.
    func prefill(_ x: [Float], rows: Int, P: Int, qk8: Bool = false) throws -> [Float] {
        let M = (rows + 63) / 64 * 64
        let xs = x + [Float](repeating: 0, count: (M - rows) * hidden)
        let xb = Syn.halfBuffer(ctx, xs), out = ctx.buffer(length: M * hidden * 4)
        let s = (
            proj: ctx.buffer(length: M * projN * 4), q: ctx.buffer(length: kvH * G * M * hd * 2),
            out: ctx.buffer(length: M * outDim * 2)
        )
        let q8 = qk8 ? (q: ctx.buffer(length: kvH * G * M * hd), s: ctx.buffer(length: kvH * G * M * 4)) : nil
        a.pos = P
        try ctx.run { e in
            try a.encodePrefill(e, x: xb, out: out, rows: rows, padded: M, s: s, qk8: q8)
        }
        return Syn.floats(out, rows * hidden)
    }
}

// MARK: - MoE of the round and the prefill

/// A MoE layer under test with its kernels bound like MoELayer (round: vnax_moe_gateup → vnax_moe_down_split →
/// vmoe_combine9) and PrefillMoE (moe_group → moe_gateup_swiglu2 → moe_down2 → moe_combine_acc[_ns[_f32]]). Each
/// matrix goes as (w, scales, biases); MXFP4 puts its e8m0 scales in the bias slot too (unread). Without a shared expert
/// its slots get an empty matrix.
struct MoEPipeline {
    let ctx: MetalContext
    let D: Int, I: Int, E: Int, K: Int
    /// Intermediate of the shared expert (0 = none).
    let S: Int
    let gate: [MTLBuffer], up: [MTLBuffer], down: [MTLBuffer]
    let sGate: [MTLBuffer], sUp: [MTLBuffer], sDown: [MTLBuffer]
    /// Expert → slot table.
    let smap: MTLBuffer
    /// Suffix of the gate+up activation instances ("" = SiLU, "_gelu", "_relu2", "_oai").
    var act = ""
    /// MXFP4 experts: the `_mxfp4` instances, and the prefill's Yd in fp32.
    var mxfp4 = false
    /// gpt-oss expert biases (gate, up, down), empty without them.
    var bias: [MTLBuffer] = []
    /// Round instance: fragments per simdgroup and simdgroups per threadgroup.
    var nf = 2, sgs = 8

    static func bufs(_ m: DenseQ) -> [MTLBuffer] { [m.w, m.s, m.b] }

    init(
        _ ctx: MetalContext, D: Int, I: Int, E: Int, K: Int,
        experts: (gate: [MTLBuffer], up: [MTLBuffer], down: [MTLBuffer]),
        shared: (gate: [MTLBuffer], up: [MTLBuffer], down: [MTLBuffer], inter: Int)? = nil, smap: MTLBuffer
    ) {
        self.ctx = ctx; self.D = D; self.I = I; self.E = E; self.K = K; self.smap = smap
        (gate, up, down) = experts
        let none = Self.bufs(DenseQ.empty(device: ctx.device, K: D))
        (sGate, sUp, sDown, S) = shared ?? (none, none, none, 0)
    }

    /// Slots of each row in the round's partials (the top-k and the shared expert).
    var slots: Int { K + (S > 0 ? 1 : 0) }

    /// VMoEParams of a round of B rows (bit 0 of word 5: accumulate into the output).
    func roundParams(rows B: Int, accumulate: Bool = false) -> [UInt32] {
        [
            UInt32(D), UInt32(I), UInt32(E), UInt32(B), UInt32(max(I, S) / (16 * nf)),
            (accumulate ? 1 : 0) | UInt32(D / (16 * nf)) << 8 | UInt32(K) << 16 | UInt32(slots) << 24,
            UInt32(S > 0 ? S : I),
        ]
    }

    /// Buffers of a round of B rows: the routing ones (`ent`, `pairs`, `args`, `inds`, from vmoe_route or set by
    /// hand), the input (half), the output (fp32) and the scratch of h and the partials.
    struct RoundBuffers {
        let x: MTLBuffer, ent: MTLBuffer, pairs: MTLBuffer, args: MTLBuffer, inds: MTLBuffer, out: MTLBuffer
        let h: MTLBuffer, part: MTLBuffer
    }

    func roundBuffers(
        x: MTLBuffer, ent: MTLBuffer, pairs: MTLBuffer, args: MTLBuffer, inds: MTLBuffer, out: MTLBuffer, rows B: Int
    ) -> RoundBuffers {
        RoundBuffers(
            x: x, ent: ent, pairs: pairs, args: args, inds: inds, out: out,
            h: ctx.buffer(length: (B * K * I + B * max(S, I)) * 2), part: ctx.buffer(length: (K + 1) * 16 * D * 4))
    }

    /// Encodes gate+up, down_split and the combination of a round with VMoEParams `mp`; the indirect arguments at
    /// offsets 0 (gate+up) and 16 (down) of `b.args`.
    func encodeRound(_ e: MTLComputeCommandEncoder, _ b: RoundBuffers, rows B: Int, params mp: [UInt32]) throws {
        let fmt = mxfp4 ? "_mxfp4" : "", tpg = MTLSize(width: 32 * sgs, height: 1, depth: 1)
        e.setComputePipelineState(try ctx.pipeline("vnax_moe_gateup\(act)\(fmt)_nf\(nf)_sg\(sgs)"))
        e.setBuffer(b.x, offset: 0, index: 0)
        for (i, m) in (gate + up + sGate + sUp).enumerated() { e.setBuffer(m, offset: 0, index: 1 + i) }
        for (i, m) in [b.ent, b.pairs, b.h].enumerated() { e.setBuffer(m, offset: 0, index: 13 + i) }
        e.setBytes(mp, length: 28, index: 16)
        e.setBuffer(smap, offset: 0, index: 17)
        if bias.count == 3 {
            e.setBuffer(bias[0], offset: 0, index: 18)
            e.setBuffer(bias[1], offset: 0, index: 19)
        }
        e.dispatchThreadgroups(indirectBuffer: b.args, indirectBufferOffset: 0, threadsPerThreadgroup: tpg)
        e.setComputePipelineState(try ctx.pipeline("vnax_moe_down_split\(fmt)_nf\(nf)_sg\(sgs)"))
        e.setBuffer(b.h, offset: 0, index: 0)
        for (i, m) in (down + sDown).enumerated() { e.setBuffer(m, offset: 0, index: 1 + i) }
        e.setBuffer(b.ent, offset: 0, index: 7)
        e.setBuffer(b.pairs, offset: 0, index: 8)
        e.setBuffer(b.part, offset: 0, index: 10)
        e.setBytes(mp, length: 28, index: 11)
        e.setBuffer(b.inds, offset: 0, index: 12)
        e.setBuffer(smap, offset: 0, index: 13)
        if bias.count == 3 { e.setBuffer(bias[2], offset: 0, index: 14) }
        e.dispatchThreadgroups(indirectBuffer: b.args, indirectBufferOffset: 16, threadsPerThreadgroup: tpg)
        e.setComputePipelineState(try ctx.pipeline("vmoe_combine9"))
        e.setBuffer(b.part, offset: 0, index: 0)
        e.setBuffer(b.out, offset: 0, index: 1)
        e.setBytes(mp, length: 28, index: 2)
        e.setBuffer(b.part, offset: 0, index: 3)
        e.dispatchThreads(
            MTLSize(width: B * D / 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Round of B rows with the routing set by hand (K experts per row in `inds`, weights `wts`; `sw`: shared expert
    /// gate per row), its union in vmoe_route's format: experts in increasing order and the shared expert last.
    func round(_ x: [Float], rows B: Int, inds: [Int], wts: [Float], sw: [Float] = []) throws -> [Float] {
        var entries = [UInt32](), pairs = [UInt32]()
        for e in 0..<E {
            let hits = (0..<(B * K)).filter { inds[$0] == e }
            if hits.isEmpty { continue }
            entries += [UInt32(e), UInt32(pairs.count / 2), UInt32(hits.count), 0]
            for p in hits { pairs += [UInt32(p / K), wts[p].bitPattern] }
        }
        if S > 0 {
            entries += [UInt32(E), UInt32(B * K), UInt32(B), 0]
            for r in 0..<B { pairs += [UInt32(r), sw[r].bitPattern] }
        }
        let mp = roundParams(rows: B), U = UInt32(entries.count / 4)
        let b = roundBuffers(
            x: Syn.halfBuffer(ctx, x), ent: ctx.buffer(entries), pairs: ctx.buffer(pairs),
            args: ctx.buffer([mp[4], U, 1, U, UInt32(D / (16 * nf)), U, 1, 0]),
            inds: ctx.buffer(inds.map { UInt32($0) }),
            out: ctx.buffer(length: B * D * 4), rows: B)
        try ctx.run { e in try encodeRound(e, b, rows: B, params: mp) }
        return Syn.floats(b.out, B * D)
    }

    /// Encodes the prefill of T rows: grouping by expert of `inds` (moe_group), gate+up, down and the weighted
    /// combination added into `out` (fp32 [T, D]); with a shared expert, its rows go after the pairs with gate `sgate`.
    func encodePrefill(
        _ e: MTLComputeCommandEncoder, x: MTLBuffer, inds: MTLBuffer, wts: MTLBuffer, sgate: MTLBuffer?,
        out: MTLBuffer, rows T: Int, tiles: PrefillMoE.Tiles
    ) throws {
        let PT = T * K, ns = S > 0 ? T : 0, yb = mxfp4 ? 4 : 2
        let offsets = ctx.buffer(length: (E + 1) * 4), rowMap = ctx.buffer(length: (PT + T) * 4)
        let pos = ctx.buffer(length: PT * 4), nt = (PT / 16 + E + T / 16 + 2) * 16
        let tilesA = ctx.buffer(length: nt), tilesB = ctx.buffer(length: nt)
        let argsA = ctx.buffer(length: 16), argsB = ctx.buffer(length: 16)
        let hp = ctx.buffer(length: (PT * I + T * S) * 2), yd = ctx.buffer(length: (PT + ns) * D * yb)
        // Down column tiles rounded up: the last one is shifted back to end at hidden (LAB_MOE_H64).
        let gp = [
            UInt32(PT), UInt32(E), UInt32(tiles.bmA), UInt32(tiles.bmB), UInt32(max(I, S) / tiles.bnA),
            UInt32((D + tiles.bnB - 1) / tiles.bnB), UInt32(ns), 1,
        ]
        let gu = [UInt32(I), UInt32(D), UInt32(K), UInt32(E), UInt32(S > 0 ? S : I), UInt32(PT)]
        let dn = [UInt32(D), UInt32(I), UInt32(K), UInt32(E), UInt32(S > 0 ? S : I), UInt32(PT)]
        let tg128 = MTLSize(width: 128, height: 1, depth: 1)
        func bind(_ m: [MTLBuffer], at i: Int) {
            for (j, b) in m.enumerated() { e.setBuffer(b, offset: 0, index: i + j) }
        }
        e.setComputePipelineState(try ctx.pipeline("moe_group"))
        bind([inds, offsets, rowMap, pos, tilesA, tilesB, argsA, argsB], at: 0)
        e.setBytes(gp, length: 32, index: 8)
        e.setBuffer(inds, offset: 0, index: 9)
        e.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        e.setComputePipelineState(try ctx.pipeline(tiles.gateUpKernel(act: act, mxfp4: mxfp4)))
        bind([x] + gate + up + [hp, rowMap, tilesA], at: 0)
        e.setBytes(gu, length: 24, index: 10)
        bind(sGate + sUp + [smap] + bias.prefix(2), at: 11)
        e.dispatchThreadgroups(indirectBuffer: argsA, indirectBufferOffset: 0, threadsPerThreadgroup: tg128)
        e.setComputePipelineState(try ctx.pipeline(tiles.downKernel(mxfp4: mxfp4)))
        bind([hp] + down + [yd, tilesB], at: 0)
        e.setBytes(dn, length: 24, index: 6)
        bind(sDown + [smap] + bias.suffix(1), at: 7)
        e.dispatchThreadgroups(indirectBuffer: argsB, indirectBufferOffset: 0, threadsPerThreadgroup: tg128)
        e.setComputePipelineState(
            try ctx.pipeline(S > 0 ? "moe_combine_acc" : mxfp4 ? "moe_combine_acc_ns_f32" : "moe_combine_acc_ns"))
        bind([yd, pos, wts], at: 0)
        if let sgate, S > 0 {
            e.setBuffer(yd, offset: PT * D * yb, index: 3)
            e.setBuffer(sgate, offset: 0, index: 4)
        }
        e.setBuffer(out, offset: 0, index: 5)
        e.setBytes([UInt32(D), UInt32(K)], length: 8, index: 6)
        e.dispatchThreadgroups(
            MTLSize(width: T, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Prefill of T rows with the routing set by hand (`sw`: shared expert gate per row), into a zero output.
    func prefill(
        _ x: [Float], rows T: Int, inds: [Int], wts: [Float], sw: [Float] = [], tiles: PrefillMoE.Tiles = .init()
    ) throws -> [Float] {
        let out = ctx.buffer([Float](repeating: 0, count: T * D))
        let xb = Syn.halfBuffer(ctx, x), ib = ctx.buffer(inds.map { UInt32($0) }), wb = ctx.buffer(wts)
        let sg = S > 0 ? ctx.buffer(sw) : nil
        try ctx.run { e in
            try encodePrefill(e, x: xb, inds: ib, wts: wb, sgate: sg, out: out, rows: T, tiles: tiles)
        }
        return Syn.floats(out, T * D)
    }
}

/// CPU reference of a MoE layer with the routing set by hand: y[r] = Σ_k w·(down_e(act(gate_e x + bg_e, up_e x + bu_e))
/// + bd_e) + sw[r]·down_s(act(gate_s x, up_s x)), with h rounded to half like the kernels; `halfGU`: gate and up
/// rounded to half before the activation (the prefill). Rows grouped by expert, one GEMM per matrix.
struct MoEReference {
    typealias FFN = (gate: [Float], up: [Float], down: [Float])
    let D: Int, I: Int, E: Int, K: Int
    /// Float weights [I, D], [I, D] and [D, I] of expert e (only called for experts some row picks).
    let expert: (Int) -> FFN
    /// Without gate (relu²) the activation only reads up.
    var gated = true
    var act: (Float, Float) -> Float = { g, u in g / (1 + exp(-g)) * u }
    /// gpt-oss expert biases ([E·I], [E·I], [E·D]).
    var bias: FFN?
    /// Shared expert and its intermediate.
    var shared: (ffn: FFN, inter: Int)?

    func callAsFunction(_ x: [Float], rows: Int, inds: [Int], wts: [Float], sw: [Float] = [], halfGU: Bool) -> [Float] {
        func ffn(_ xs: [Float], _ m: Int, _ w: FFN, _ n: Int, bias e: Int?) -> [Float] {
            let uu = CPU.matmulT(xs, w.up, M: m, N: n, K: D)
            let gg = gated ? CPU.matmulT(xs, w.gate, M: m, N: n, K: D) : uu
            let h = (0..<(m * n)).map { i -> Float in
                var a = gg[i], b = uu[i]
                if let e, let bias {
                    a += bias.gate[e * n + i % n]
                    b += bias.up[e * n + i % n]
                }
                if halfGU { a = Float(Float16(a)); b = Float(Float16(b)) }
                return Float(Float16(act(a, b)))
            }
            var o = CPU.matmulT(h, w.down, M: m, N: D, K: n)
            if let e, let bias { for i in o.indices { o[i] += bias.down[e * D + i % D] } }
            return o
        }
        var y = [Float](repeating: 0, count: rows * D)
        if let shared {
            y = ffn(x, rows, shared.ffn, shared.inter, bias: nil)
            for r in 0..<rows { for c in 0..<D { y[r * D + c] *= sw[r] } }
        }
        for e in 0..<E {
            let hits = (0..<(rows * K)).filter { inds[$0] == e }
            if hits.isEmpty { continue }
            let o = ffn(hits.flatMap { x[($0 / K * D)..<($0 / K * D + D)] }, hits.count, expert(e), I, bias: e)
            for (j, p) in hits.enumerated() { for c in 0..<D { y[p / K * D + c] += wts[p] * o[j * D + c] } }
        }
        return y
    }
}
