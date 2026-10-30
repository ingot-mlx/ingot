import Foundation
import Metal

@testable import MetalLab

/// The router kernels against mlx-lm's own router code (ref/router_vectors.json, written by ref/router_ref.py on random
/// logits: the routers of LFM2.5, GLM-4.7-Flash, gpt-oss, Gemma 4 26B-A4B and Nemotron-H). Each case compiles its own
/// library (E, top-k, scoring, bias, scale and epsilon as macros) and runs the round's and the prefill's route kernels
/// that MoEKernelSet picks for its RouterSpec.
/// Rows 0-1 carry an exact tie for the k-th place: the engine keeps the lower index (checked against the CPU), while
/// mlx-lm's argpartition keeps either (the higher one with `kth=-k`, lfm2_moe, gpt_oss; the lower
/// one with `-scores, kth=k-1`, group_expert_select), so only rows 2… are compared with mlx-lm.
struct RouterVector: Sendable, Decodable {
    let name: String, experts: Int, topK: Int, score: String, scale: Float, eps: Float, rows: Int
    let logits: [Float], bias: [Float]?, weights: [Float]
    /// Router linear bias (gpt-oss `mlp.router.bias`, LAB_MOE_LBIAS): added to the logits before top-k and weights.
    let logitBias: [Float]?
    /// Per-expert scale (Gemma 4 `router.per_expert_scale`, LAB_MOE_PES): multiplies each chosen expert's weight after
    /// the renormalization.
    let perExpertScale: [Float]?
    var sigmoid: Bool { score == "sigmoid" }

    private struct File: Decodable { let cases: [RouterVector] }

    static let all: [RouterVector] = {
        let url = ModelSpecTests.repo.appendingPathComponent("ref/router_vectors.json")
        guard let d = try? Data(contentsOf: url), let f = try? JSONDecoder().decode(File.self, from: d) else {
            preconditionFailure("ref/router_vectors.json: missing or unreadable (ref/router_ref.py writes it)")
        }
        return f.cases
    }()

    static func named(_ n: String) -> RouterVector { all.first { $0.name == n }! }

    var spec: RouterSpec {
        var s = RouterSpec(
            scoring: sigmoid ? .sigmoid : .softmax, shared: .none, experts: experts, topK: topK, inter: 512,
            sharedInter: 0, selectBias: bias != nil, scale: scale, normEps: eps)
        s.logitBias = logitBias != nil
        s.perExpertScale = perExpertScale != nil
        return s
    }

    /// The library of the case: Qwen3.6's shape with this router.
    var dims: ModelDims {
        var d = ModelDims()
        d.moeE = experts; d.moeK = topK; d.moeShared = 0
        d.moeSigmoid = sigmoid; d.moeBias = bias != nil; d.moeRScale = scale; d.moeEps = eps
        d.moeLBias = logitBias != nil
        d.moePES = perExpertScale != nil
        return d
    }

    /// The engine's rule on the CPU: top-k by score (+ selection bias), ties to the lower index; weights from the
    /// unbiased scores (softmax over the top-k, or the sigmoid p), renormalized, then scaled (and by the per-expert scale).
    /// A router linear bias is part of the logits.
    func reference() -> (inds: [Float], wts: [Float]) {
        let E = experts, K = topK
        var inds = [Float](), wts = [Float]()
        for r in 0..<rows {
            let l = (0..<E).map { logits[r * E + $0] + (logitBias?[$0] ?? 0) }
            let p = sigmoid ? l.map { 1 / (1 + exp(-$0)) } : l
            let sel = (0..<E).map { p[$0] + (bias?[$0] ?? 0) }
            let top = Array((0..<E).sorted { sel[$0] > sel[$1] || (sel[$0] == sel[$1] && $0 < $1) }.prefix(K))
            var w: [Float]
            if sigmoid {
                w = top.map { p[$0] }
                let s = w.reduce(0, +) + eps
                w = w.map { $0 / s }
            } else {
                let m = l[top[0]]
                w = top.map { exp(l[$0] - m) }
                let s = w.reduce(0, +)
                w = w.map { $0 / s }
            }
            inds += top.map { Float($0) }
            wts += zip(top, w).map { e, v in v * scale * (perExpertScale?[e] ?? 1) }
        }
        return (inds, wts)
    }
}

extension KernelCase {
    /// One case per mlx-lm vector whose router has kernels (MoEKernelSet).
    static var routerMLX: [KernelCase] {
        RouterVector.all.filter { (try? MoEKernelSet.forRouter($0.spec, inter: 512)) != nil }.map { v in
            KernelCase(
                id: "router-mlx-\(v.name)", model: "router:\(v.name)",
                catches: "router against mlx-lm (\(v.name)): indices and weights of the round and the prefill with E = "
                    + "\(v.experts), top-\(v.topK), \(v.sigmoid ? "sigmoid" : "softmax"), bias \(v.bias != nil), "
                    + "router linear bias \(v.logitBias != nil), scale \(v.scale), per-expert scale "
                    + "\(v.perExpertScale != nil), with the logits at the stride padded to 64"
            ) { ctx, _ in try routerChecks(ctx, RouterVector.named(v.name)) }
        }
    }

    static func routerChecks(_ ctx: MetalContext, _ v: RouterVector) throws -> [Check] {
        let E = v.experts, K = v.topK, B = v.rows, D = 2048
        _ = try MoEKernelSet.forRouter(v.spec, inter: 512)  // vmoe_route and moe_route_topk_cfg for every router
        let ld = MoELayer.routerLD(E)
        var padded = [Float](repeating: 0, count: B * ld)
        for r in 0..<B { padded.replaceSubrange((r * ld)..<(r * ld + E), with: v.logits[(r * E)..<(r * E + E)]) }
        let lb = ctx.buffer(padded), bb = ctx.buffer(v.bias ?? v.logitBias ?? [Float](repeating: 0, count: E))
        let pes = ctx.buffer(v.perExpertScale ?? [Float](repeating: 1, count: E))
        let z = ctx.buffer(length: D * 2), ent = ctx.buffer(length: (B * K + 1) * 16)
        let pairs = ctx.buffer(length: (B * K + B) * 8), args = ctx.buffer(length: 64)
        let ib = ctx.buffer(length: B * K * 4), wb = ctx.buffer(length: B * K * 4)
        let ib2 = ctx.buffer(length: B * K * 4), wb2 = ctx.buffer(length: B * K * 4), sg2 = ctx.buffer(length: B * 4)
        try ctx.run { e in
            try encodeRouteCommon(
                ctx, e, logits: lb, bias: bb, zero: z, rowLimit: args, outs: [ent, pairs, args, ib, wb],
                params: [UInt32(D), 512, UInt32(E), UInt32(B), 16, UInt32(K) << 16 | UInt32(K) << 24, 512], pes: pes)
            try encodePrefillRouteCommon(
                ctx, e, logits: lb, bias: bb, zero: z, outs: [ib2, wb2, sg2],
                params: [UInt32(B), UInt32(E), UInt32(D), UInt32(K)], pes: pes)
        }
        let ref = v.reference()
        // Dense [rows 2…, E] weights of the kernel, against mlx-lm's.
        func dense(_ ib: MTLBuffer, _ wb: MTLBuffer) -> [Float] {
            let i = Syn.uints(ib, B * K), w = Syn.floats(wb, B * K)
            var out = [Float](repeating: 0, count: (B - 2) * E)
            for r in 2..<B { for k in 0..<K { out[(r - 2) * E + Int(i[r * K + k])] += w[r * K + k] } }
            return out
        }
        let mlx = Array(v.weights[(2 * E)...])
        var checks: [Check] = []
        for (tag, i, w) in [("round", ib, wb), ("prefill", ib2, wb2)] {
            let d = dense(i, w)
            checks += [
                Check(
                    label: "\(tag): experts (CPU, ties to the lower index)", gpu: Syn.uints(i, B * K).map { Float($0) },
                    ref: ref.inds, tol: .bitExact),
                Check(label: "\(tag): weights (CPU)", gpu: Syn.floats(w, B * K), ref: ref.wts, tol: .maxRel(1e-5)),
                Check(
                    label: "\(tag): chosen experts (mlx-lm, rows 2…)", gpu: d.map { $0 != 0 ? 1 : 0 },
                    ref: mlx.map { $0 != 0 ? 1 : 0 }, tol: .bitExact),
                Check(label: "\(tag): weights (mlx-lm, rows 2…)", gpu: d, ref: mlx, tol: .maxRel(1e-5)),
            ]
        }
        return checks
    }
}

extension KernelCase {
    /// Round route kernel (vmoe_route) as MoELayer.encode binds it, without the input rows and the shared gate (buffers
    /// 1–4 get `zero`): logits at 0, `outs` (entries, pairs, args, inds, wts) at 5–9, the VMoEParams words at 10, the
    /// round state at 11 (read only with flags bit 1), the selection bias at 12 and the per-expert scale at 13 (read only
    /// with LAB_MOE_PES). One threadgroup of 512 threads.
    static func encodeRouteCommon(
        _ ctx: MetalContext, _ e: MTLComputeCommandEncoder, logits: MTLBuffer, bias: MTLBuffer, zero: MTLBuffer,
        rowLimit: MTLBuffer, outs: [MTLBuffer], params: [UInt32], pes: MTLBuffer? = nil
    ) throws {
        e.setComputePipelineState(try ctx.pipeline("vmoe_route"))
        for (i, b) in [logits, zero, zero, zero, zero].enumerated() { e.setBuffer(b, offset: 0, index: i) }
        for (i, b) in outs.enumerated() { e.setBuffer(b, offset: 0, index: 5 + i) }
        e.setBytes(params, length: 4 * params.count, index: 10)
        e.setBuffer(rowLimit, offset: 0, index: 11)
        e.setBuffer(bias, offset: 0, index: 12)
        e.setBuffer(pes ?? bias, offset: 0, index: 13)
        e.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
    }

    /// Prefill top-k kernel (moe_route_topk_cfg) as PrefillMoE binds it, without the input rows and the shared gate:
    /// `outs` (inds, wts, shared gate) at 5–7, RouteParams (T, E, D, topk) at 8, the selection bias at 9 and the
    /// per-expert scale at 10 (LAB_MOE_PES); 4 rows per threadgroup of 128 threads.
    static func encodePrefillRouteCommon(
        _ ctx: MetalContext, _ e: MTLComputeCommandEncoder, logits: MTLBuffer, bias: MTLBuffer, zero: MTLBuffer,
        outs: [MTLBuffer], params: [UInt32], pes: MTLBuffer? = nil
    ) throws {
        e.setComputePipelineState(try ctx.pipeline("moe_route_topk_cfg"))
        for (i, b) in [logits, zero, zero, zero, zero].enumerated() { e.setBuffer(b, offset: 0, index: i) }
        for (i, b) in outs.enumerated() { e.setBuffer(b, offset: 0, index: 5 + i) }
        e.setBytes(params, length: 4 * params.count, index: 8)
        e.setBuffer(bias, offset: 0, index: 9)
        e.setBuffer(pes ?? bias, offset: 0, index: 10)
        e.dispatchThreadgroups(
            MTLSize(width: (Int(params[0]) + 3) / 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
}
