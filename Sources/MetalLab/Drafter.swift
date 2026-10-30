import Foundation
import Metal

// DFlash2 drafter for Qwen3.6 (`incoai/Qwen3.6-35B-A3B-DFlash2`: 6 layers, hidden 2048, context of 8 layers) or for the
// 27B (the `drafter/` of the model pack, `incoai/Qwen3.8-27B-DFlash2`: 5 layers, hidden 5120, FFN 17408, context of 5
// layers): dense qwen3 layers with dynamic conv, windowed context of 2048 taken from big-model layers, non-causal
// block of 8 or 16 (anchor + masks; Muse Glimmer's is 16), lm_head of the big model and candidate selector. Protocol and shape after mlx-serve
// (`src/dflash.zig`, MIT) and splash (Apache-2.0, quantizes the drafter to 4 bits at load). Also DFlash v1 (`v1`), Gemma 4
// 26B-A4B's default (`z-lab/gemma-4-26B-A4B-it-DFlash`: 5 layers, hidden 2816, context of 6 layers), and gpt-oss 20b's
// (`z-lab/gpt-oss-20b-DFlash`: 8 layers, 64/8 heads of 64 with q/k/v/o biases, YaRN RoPE, context of 5 layers).
// Kernels: yy_model.metal.

struct DAttnParams {
    var P: UInt32, nvalid: UInt32, pos0: UInt32, C: UInt32, nsplit: UInt32
    var scale: Float, eps: Float, theta: Float
    var dyn: UInt32 = 0
    var block: UInt32 = 8
    var win = UInt32(DFlash2Drafter.window), slots = UInt32(DFlash2Drafter.slots)
}

public final class DFlash2Drafter {
    /// Blocks the kernels support (anchor + block − 1 masks): 8 (Qwen3.6, Flash-Next) or 16 (Muse Glimmer), from the
    /// drafter's `dflash_config.block_size`. The slots hold the window plus the largest block, to a multiple of 32.
    public static let blocks: Set<Int> = [8, 16]
    public static let window = 2048, slots = 2080
    /// Window and slots of the full-attention layer of a DFlash v1 drafter (Gemma 4 26B-A4B's last one): a fixed ring of
    /// 8192 instead of 2048 (MP-22 T4b: τ ×1.04 in MLX; the memory does not grow with the context).
    public static let fullWindow = 8192, fullSlots = 8224
    /// Context window of each layer of a drafter with config `cj`: `fullWindow` for the v1 layers that `layer_types`
    /// marks `full_attention`, `window` for the rest (all of DFlash2's).
    static func layerWindows(config cj: [String: Any], layers: Int, v1: Bool) -> [Int] {
        let types = cj["layer_types"] as? [String] ?? []
        return (0..<layers).map { v1 && $0 < types.count && types[$0] == "full_attention" ? fullWindow : window }
    }
    static func slots(window w: Int) -> Int { w == fullWindow ? fullSlots : Self.slots }
    /// The widest context window of the layers: the prefill writes that many last positions of each call.
    public var maxWindow: Int { layers.map(\.window).max() ?? Self.window }
    /// Context margin the engine keeps past a prompt: two drafter blocks (16 without a drafter, as with block 8).
    public static func margin(_ dr: DFlash2Drafter?) -> Int { 2 * (dr?.block ?? 8) }
    /// Block of this drafter (rows of each draft pass; the big model verifies up to that many).
    public let block: Int
    /// Draft logits post-processing before the selector (Muse Glimmer's DFlash2: `dflash_config.output_multiplier`,
    /// then `final_logit_softcapping`, as the reference): applied to the top-16 values (monotonic, same candidates).
    /// nil: raw logits (Qwen3.6, Flash-Next).
    let logitsPost: (mul: Float, cap: Float)?
    /// Bits of the drafter quantized at load: 4 (at 8 bits it accepts 1.4% more and costs 1.3 ms more per round).
    static let bits = 4
    public let ctxWidth: Int
    public struct Layer {
        public let n1, n2, qn, kn: MTLBuffer
        /// Attention biases (bf16) of a drafter with `attention_bias`: q | k | v (the k | v part also feeds the
        /// context) and o_proj; nil without them.
        public let qkvBias, oBias: MTLBuffer?
        /// Dynamic conv of DFlash2 (nil in DFlash v1).
        public let abase, mbase: MTLBuffer?
        public let qkv, kv, o, gu, down: DenseQ
        public let aproj, mproj: DenseQ?
        public let kc, vt: MTLBuffer
        /// Context window and ring slots of this layer (kc [8, slots, head dim], vt [8, head dim, slots] half).
        public let window, slots: Int
    }
    /// DFlash v1 (z-lab/dflash `DFlashDraftModel` in `dflash/model.py`): the same qwen3 layers with the big model's context, but without
    /// dynamic conv or selector: the draft of each position is the argmax of its logits. The context window
    /// stays at 2048 in the sliding layers and is a fixed ring of 8192 in the full-attention one (z-lab's uses 4096
    /// and full attention: an approximation for long contexts).
    public let v1: Bool

    public let ctx: MetalContext
    public let targetLayers: [Int]
    public let maskToken: UInt32
    public let eps: Float, theta: Float
    /// Attention shape: head dim and q heads over 8 KV heads; 128 and 32 (Qwen3.6, Gemma 4) or 64 and 64 (gpt-oss's,
    /// with biases and YaRN), the two instances of the kernels (`DFlash2Drafter.mismatch`).
    public let headDim: Int, qHeads: Int
    /// RoPE table (inv_freq per pair + mscale, RopeTable) of a drafter with a scaled RoPE (YaRN); nil: θ on the GPU.
    let ropeTab: MTLBuffer?
    /// Own lm_head for the drafts when the big model's is 8 bits: the same matrix requantized to 4 bits at load
    /// (affine, groups of 64), as cheap as a 4-bit head; the verification keeps the big model's (gpt-oss: 4.1 → ~2.2 ms
    /// per round, τ(6) −1.2% in MLX). nil: the big model's head.
    public let draftHead: DenseQ?
    public let layers: [Layer]
    public let fc: DenseQ
    public let hproj: DenseQ?
    public let hnorm, fnorm: MTLBuffer
    public let pred, succ: MTLBuffer?
    public let capRows: Int
    /// Captures of the big model [capRows, nTargets·H] half (layer j in columns [H j, +H)). A prefill group with
    /// two chunks with a drafter swaps this buffer for another while it encodes the context of the second.
    public var caps: MTLBuffer
    let fcout, enc, kvout: MTLBuffer
    let x, xn, a, dyn, qkv, q, opart, ml, attn, y, gu, hb, hid, hp, cidx, cval: MTLBuffer
    public let logits: MTLBuffer
    /// Confidence of each draft: softmax(s/τ) of the candidate chosen by `dselect` (v1: `dpick_top1`) over the top-16,
    /// τ = confTau.
    public let conf: MTLBuffer
    public static let confTau: Float = 1.5
    /// Tokens of the drafter block: [anchor, mask × (block − 1)].
    public let dtok: MTLBuffer
    static let chunkKeys = 256
    /// Key chunks of the attention of layer `l`: 9 for the 2080 slots, 33 for the 8224 ones.
    static func splits(_ l: Layer) -> Int { (l.slots + chunkKeys - 1) / chunkKeys }
    var vq: VQMV
    public let loadSeconds: Double
    /// hidden and intermediate size of the drafter.
    public let H: Int, inter: Int

    /// Multiplier of the block embedding (fp32, after the lookup): the big model's `embedScale` for DFlash v1 (z-lab's
    /// MLX `bind` multiplies by the target's `embed_scale`: √2816 in Gemma 4 26B-A4B, where leaving it out costs ~10% of
    /// τ at T = 1), 1 for DFlash2.
    public let embedScale: Float

    /// `embedScale`: the big model's embedding multiplier, applied to the block only with a DFlash v1 drafter.
    /// `lmHead`: the big model's; an 8-bit one gets a 4-bit copy for the drafts (`draftHead`).
    public init(
        ctx: MetalContext, directory: URL, vocab: Int, capRows: Int = 2048, embedScale: Float = 1,
        lmHead: DenseQ? = nil
    ) throws {
        let t0 = Date()
        self.ctx = ctx
        let d = ctx.device
        let store = try SafetensorsStore(directory: directory)
        func bad(_ what: String) -> SafetensorsError { .format("drafter config.json: missing \(what)") }
        guard
            let cj = try JSONSerialization.jsonObject(
                with: Data(contentsOf: directory.appendingPathComponent("config.json"))) as? [String: Any],
            let dc = cj["dflash_config"] as? [String: Any]
        else { throw bad("dflash_config") }
        func int(_ d: [String: Any], _ k: String) throws -> Int {
            guard let v = (d[k] as? NSNumber)?.intValue else { throw bad(k) }
            return v
        }
        guard let layerIds = dc["target_layer_ids"] as? [NSNumber] else { throw bad("target_layer_ids") }
        targetLayers = layerIds.map(\.intValue)
        maskToken = UInt32(try int(dc, "mask_token_id"))
        let mul = Float((dc["output_multiplier"] as? NSNumber)?.doubleValue ?? 1)
        let cap = Float((dc["final_logit_softcapping"] as? NSNumber)?.doubleValue ?? 0)
        logitsPost = mul != 1 || cap > 0 ? (mul, cap) : nil
        eps = Float((cj["rms_norm_eps"] as? NSNumber)?.doubleValue ?? 1e-6)
        theta = RopeSpec.theta(cj)
        headDim = (cj["head_dim"] as? NSNumber)?.intValue ?? 128
        qHeads = try int(cj, "num_attention_heads")
        let nl = try int(cj, "num_hidden_layers")
        H = try int(cj, "hidden_size")
        inter = try int(cj, "intermediate_size")
        ctxWidth = targetLayers.count * H
        v1 = store.tensors[Self.v2Tensor] == nil
        self.embedScale = v1 ? embedScale : 1
        // DFlash v1 keeps the block of 8 it always ran with; DFlash2 takes its block_size (checked by `mismatch`).
        block = v1 ? 8 : try int(dc, "block_size")
        // The same check as the drafter resolution (DrafterFit.swift); the target layers are checked there, with the
        // model's config.
        if let why = Self.mismatch(config: cj, hidden: ctx.dims.hidden, layers: nil, v1: v1) {
            throw SafetensorsError.format("DFlash2 drafter with an unsupported shape (\(why))")
        }
        // A scaled RoPE (only the head-64 shape, `mismatch`): the table the kernels read instead of θ.
        let scaling = try Self.ropeScaling(cj)
        let th = Double(theta), hd0 = headDim
        ropeTab = scaling.map { ctx.buffer(RopeTable.make(theta: th, rotaryDims: hd0, scaling: $0)) }
        let biased = cj["attention_bias"] as? Bool ?? false
        // bf16 weights quantized to 4 bits at load (as splash; at 8 bits it accepts 1.4% more and costs 1.3 ms more).
        func t(_ n: String) throws -> MTLBuffer { try alignedTensor(store, n, d) }
        // Norms read by rmsnorm_rows (type LAB_NWT): with a small-weights fp32 model (Bonsai), bf16 → fp32 (exact).
        func nw(_ n: String) throws -> MTLBuffer {
            guard ctx.dims.normF32 else { return try t(n) }
            let i = try store.info(n)
            guard i.dtype == .bf16 else { throw SafetensorsError.format("\(n): expected bf16") }
            return ctx.buffer(store.array(i, as: UInt16.self).map { bf16ToFloat($0) })
        }
        func q4(_ ns: [String]) throws -> DenseQ {
            try DenseQ.quantize4(store: store, names: ns.map { $0 + ".weight" }, device: d, bits: Self.bits)
        }
        // Biases the config asks for (`attention_bias`): a load error if missing. q | k | v concatenated in bf16.
        func biases(_ names: [String]) throws -> MTLBuffer {
            var all: [UInt16] = []
            for n in names {
                guard let i = store.tensors[n] else {
                    throw SafetensorsError.format("drafter config.json asks for attention biases but \(n) is missing")
                }
                guard i.dtype == .bf16, i.shape.count == 1 else {
                    throw SafetensorsError.format("\(n): expected bf16 [N]")
                }
                all += store.array(i, as: UInt16.self)
            }
            return ctx.buffer(all)
        }
        var ls: [Layer] = []
        let wins = Self.layerWindows(config: cj, layers: nl, v1: v1)
        let ring = 8 * headDim * 2
        for i in 0..<nl {
            let p = "layers.\(i)."
            let S = Self.slots(window: wins[i])
            let kc = ctx.buffer(length: S * ring), vt = ctx.buffer(length: S * ring)
            memset(kc.contents(), 0, kc.length); memset(vt.contents(), 0, vt.length)
            let a = p + "self_attn."
            ls.append(
                Layer(
                    n1: try nw(p + "input_layernorm.weight"), n2: try nw(p + "post_attention_layernorm.weight"),
                    qn: try t(p + "self_attn.q_norm.weight"), kn: try t(p + "self_attn.k_norm.weight"),
                    qkvBias: biased ? try biases(["q_proj", "k_proj", "v_proj"].map { a + $0 + ".bias" }) : nil,
                    oBias: biased ? try biases([a + "o_proj.bias"]) : nil,
                    abase: v1 ? nil : try t(p + "attention_conv.base_kernel"),
                    mbase: v1 ? nil : try t(p + "mlp_conv.base_kernel"),
                    qkv: try q4(["q_proj", "k_proj", "v_proj"].map { p + "self_attn." + $0 }),
                    kv: try q4(["k_proj", "v_proj"].map { p + "self_attn." + $0 }),
                    o: try q4([p + "self_attn.o_proj"]),
                    gu: try q4([p + "mlp.gate_proj", p + "mlp.up_proj"]),
                    down: try q4([p + "mlp.down_proj"]),
                    aproj: v1 ? nil : try q4([p + "attention_conv.kernel_projection"]),
                    mproj: v1 ? nil : try q4([p + "mlp_conv.kernel_projection"]),
                    kc: kc, vt: vt, window: wins[i], slots: S))
        }
        layers = ls
        fc = try q4(["fc"])
        hnorm = try nw("hidden_norm.weight")
        fnorm = try nw("norm.weight")
        if v1 {
            hproj = nil; pred = nil; succ = nil
        } else {
            hproj = try q4(["candidate_selector.hidden_projection"])
            // The codebooks may or may not carry the .weight suffix depending on the converter (naklitechie's does).
            let cb = store.tensors["candidate_selector.predecessor_codebook"] != nil ? "" : ".weight"
            pred = try t("candidate_selector.predecessor_codebook" + cb)
            succ = try t("candidate_selector.successor_codebook" + cb)
            guard try store.info("candidate_selector.predecessor_codebook" + cb).shape == [vocab, 256] else {
                throw SafetensorsError.format("selector codebooks with another vocabulary")
            }
        }
        self.capRows = capRows
        caps = ctx.buffer(length: capRows * ctxWidth * 2)
        fcout = ctx.buffer(length: capRows * H * 4)
        enc = ctx.buffer(length: capRows * H * 2)
        let hd = headDim, qh = qHeads
        kvout = ctx.buffer(length: capRows * 16 * hd * 4)  // k (8 × head dim) | v (8 × head dim)
        let B = block
        x = ctx.buffer(length: B * H * 4); xn = ctx.buffer(length: B * H * 2); a = ctx.buffer(length: B * H * 2)
        dyn = ctx.buffer(length: B * (4 * H / 16) * 4); qkv = ctx.buffer(length: B * (qh + 16) * hd * 4)
        q = ctx.buffer(length: qh * B * hd * 2)  // qh / 8 query heads per KV head and row
        let ns = ls.map { Self.splits($0) }.max()!
        opart = ctx.buffer(length: ns * qh * B * hd * 4); ml = ctx.buffer(length: ns * qh * B * 8)
        attn = ctx.buffer(length: B * qh * hd * 2); y = ctx.buffer(length: B * H * 4)
        gu = ctx.buffer(length: B * 2 * inter * 4); hb = ctx.buffer(length: B * inter * 2)
        hid = ctx.buffer(length: B * H * 2)
        hp = ctx.buffer(length: B * 256 * 4); cidx = ctx.buffer(length: B * 16 * 4)
        cval = ctx.buffer(length: B * 16 * 4)
        conf = ctx.buffer(length: Self.blocks.max()! * 4)
        logits = ctx.buffer(length: (B - 1) * vocab * 4)
        dtok = ctx.buffer(length: B * 4)
        draftHead = try lmHead.flatMap { h in
            guard h.bits == 8 else { return nil }
            guard h.gs == 64 && h.K % 64 == 0 else {
                throw SafetensorsError.format("8-bit lm_head without groups of 64")
            }
            return DenseQ.requantize(h, bits: 4, device: d)
        }
        vq = VQMV(ctx)
        loadSeconds = Date().timeIntervalSince(t0)
    }

    public var weightBytes: Int {
        layers.reduce(
            fc.bytes + (hproj?.bytes ?? 0) + (pred?.length ?? 0) + (succ?.length ?? 0) + (draftHead?.bytes ?? 0)
        ) {
            $0 + $1.qkv.bytes + $1.o.bytes + $1.gu.bytes + $1.down.bytes + ($1.aproj?.bytes ?? 0)
                + ($1.mproj?.bytes ?? 0)
        }
    }

    private func params(_ l: Layer, P: Int = 0, nvalid: Int = 0, pos0: Int = 0) -> DAttnParams {
        DAttnParams(
            P: UInt32(P), nvalid: UInt32(nvalid), pos0: UInt32(pos0), C: UInt32(Self.chunkKeys),
            nsplit: UInt32(Self.splits(l)), scale: 1 / Float(headDim).squareRoot(), eps: eps, theta: theta,
            block: UInt32(block), win: UInt32(l.window), slots: UInt32(l.slots))
    }

    private func norm(
        _ e: MTLComputeCommandEncoder, _ src: MTLBuffer, _ w: MTLBuffer, _ dst: MTLBuffer, rows: Int,
        half: Bool = false
    ) throws {
        var p = (UInt32(H), eps, UInt32(H), UInt32(H))
        e.setComputePipelineState(try ctx.pipeline(half ? "rmsnorm_rows_h" : "rmsnorm_rows"))
        e.setBuffer(src, offset: 0, index: 0)
        e.setBuffer(w, offset: 0, index: 1)
        e.setBuffer(dst, offset: 0, index: 2)
        e.setBytes(&p, length: 16, index: 3)
        e.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    private func lin(
        _ e: MTLComputeCommandEncoder, x: MTLBuffer, xOffset: Int = 0, _ m: DenseQ, y: MTLBuffer, rows: Int,
        accumulate: Bool = false
    ) throws {
        if rows <= 16 {
            try vq.encode(e, x: x, xOffset: xOffset, m, y: y, rows: rows, accumulate: accumulate)
        } else {
            try m.encodeQmmF32(ctx, e, x: x, xOffset: xOffset, y: y, rows: rows)
        }
    }

    /// Context: `rows` rows of `caps` (the first `nvalid` are valid, positions pos0...) -> fc -> hidden_norm -> K/V of
    /// each layer in the window ring. With rows > 16, rows must be a multiple of 64 (prefill).
    /// fnc: for each of `ringAt`, also the ring as it stands after its first `rows` rows (a capture inside
    /// the pass, as if it ended there) in its `targets`: copy of the earlier ring and the same writes.
    public func encodeContext(
        _ e: MTLComputeCommandEncoder, rows: Int, nvalid: Int, pos0: Int, state: MTLBuffer? = nil,
        ringAt: [(rows: Int, targets: [(kc: MTLBuffer, vt: MTLBuffer)])] = []
    ) throws {
        precondition(rows <= capRows && nvalid <= rows)
        precondition(ringAt.allSatisfy { $0.rows > 0 && $0.rows <= nvalid && $0.targets.count == layers.count })
        try lin(e, x: caps, fc, y: fcout, rows: rows)
        try norm(e, fcout, hnorm, enc, rows: rows)
        func write(_ kc: MTLBuffer, _ vt: MTLBuffer, _ l: Layer, rows n: Int) throws {
            var p = params(l, nvalid: nvalid, pos0: pos0)
            p.dyn = state == nil ? 0 : 1
            e.setComputePipelineState(try ctx.pipeline("dctx_write" + kernelSuffix))
            for (i, b) in [kvout, l.kn, kc, vt].enumerated() { e.setBuffer(b, offset: 0, index: i) }
            e.setBytes(&p, length: MemoryLayout<DAttnParams>.size, index: 4)
            e.setBuffer(state ?? kvout, offset: 0, index: 5)
            // k | v biases: past the q part of q | k | v.
            e.setBuffer(l.qkvBias ?? kvout, offset: l.qkvBias == nil ? 0 : qHeads * headDim * 2, index: 6)
            e.setBuffer(ropeTab ?? kvout, offset: 0, index: 7)
            e.dispatchThreadgroups(
                MTLSize(width: 16, height: n, depth: 1),
                threadsPerThreadgroup: MTLSize(width: headDim, height: 1, depth: 1))
        }
        for (j, l) in layers.enumerated() {
            try lin(e, x: enc, l.kv, y: kvout, rows: rows)
            for r in ringAt {
                let t = r.targets[j]
                try copyRing(e, l.kc, t.kc)
                try copyRing(e, l.vt, t.vt)
                try write(t.kc, t.vt, l, rows: r.rows)
            }
            try write(l.kc, l.vt, l, rows: nvalid)
        }
    }

    /// Suffix of the attention kernels of this drafter's head (yy_model.metal `DATTN_INST`).
    var kernelSuffix: String { headDim == 64 ? "_h64" : "" }

    /// Whole (bit-exact) copy of a ring buffer.
    private func copyRing(_ e: MTLComputeCommandEncoder, _ src: MTLBuffer, _ dst: MTLBuffer) throws {
        precondition(dst.length == src.length && src.length % 16 == 0)
        var n4 = UInt32(src.length / 16)
        e.setComputePipelineState(try ctx.pipeline("copy_u32"))
        e.setBuffer(src, offset: 0, index: 0)
        e.setBuffer(dst, offset: 0, index: 1)
        e.setBytes(&n4, length: 4, index: 2)
        e.dispatchThreads(
            MTLSize(width: Int(n4), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    private func conv(_ e: MTLComputeCommandEncoder, finish: Bool, _ src: MTLBuffer, base: MTLBuffer, _ dst: MTLBuffer)
        throws
    {
        e.setComputePipelineState(try ctx.pipeline(finish ? "dconv_fin_add" : "dconv_prep"))
        for (i, b) in [src, dyn, base, dst].enumerated() { e.setBuffer(b, offset: 0, index: i) }
        e.dispatchThreads(
            MTLSize(width: block * H, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Block [anchor, mask × (block − 1)] at position P (the CPU writes `dtok`) -> drafts in `vtok[1..<block]` (the big model
    /// has the anchor in vtok[0]). `embed` and `lmHead` are those of the big model.
    /// Tree verification (`tree`): instead of the selector chain, builds a `nodes`-row tree on the GPU (dtree_build)
    /// With `roundCut` (`--round-trace 2`), each piece starts a new encoder; returns the last one (without it, `enc`).
    @discardableResult
    public func encodeBlock(
        _ enc: MTLComputeCommandEncoder?, P: Int, embed: DenseQ, lmHead: DenseQ, vtok: MTLBuffer,
        state: MTLBuffer? = nil, timeline tl: GPUTimeline? = nil, tree: (buf: MTLBuffer, nodes: Int)? = nil,
        roundCut: ((MTLComputeCommandEncoder, String) -> MTLComputeCommandEncoder)? = nil
    ) throws -> MTLComputeCommandEncoder? {
        // With `timeline`, each piece goes in its segment of the timeline (profile); otherwise, all in `enc`.
        var cur = enc
        func S(_ name: String, _ f: (MTLComputeCommandEncoder) throws -> Void) throws {
            if let tl {
                try tl.seg("draft." + name, f)
            } else if let roundCut {
                cur = roundCut(cur!, "dr." + name)
                try f(cur!)
            } else {
                try f(enc!)
            }
        }
        let st = state ?? dtok
        let B = block, tgHD = MTLSize(width: headDim, height: 1, depth: 1), G = qHeads / 8
        try S("embed") { e in
            try embed.encodeEmbed(ctx, e, tok: dtok, outF: x, rows: B)
            if embedScale != 1 {
                var sp = (embedScale, UInt32(B * H))
                e.setComputePipelineState(try ctx.pipeline("scale_rows"))
                e.setBuffer(x, offset: 0, index: 0)
                e.setBytes(&sp, length: 8, index: 1)
                e.dispatchThreads(
                    MTLSize(width: B * H, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
        }
        for l in layers {
            var p = params(l, P: P)
            p.dyn = state == nil ? 0 : 1
            try S("norm") { try norm($0, x, l.n1, xn, rows: B) }
            if let ap = l.aproj, let ab = l.abase {
                try S("convproj") { try lin($0, x: xn, ap, y: dyn, rows: B) }
                try S("conv") { try conv($0, finish: false, xn, base: ab, a) }
            }
            try S("qkv") { try lin($0, x: v1 ? xn : a, l.qkv, y: qkv, rows: B) }
            try S("attn") { e in
                e.setComputePipelineState(try ctx.pipeline("dattn_prework" + kernelSuffix))
                for (i, b) in [qkv, l.qn, l.kn, q, l.kc, l.vt].enumerated() { e.setBuffer(b, offset: 0, index: i) }
                e.setBytes(&p, length: MemoryLayout<DAttnParams>.size, index: 6)
                e.setBuffer(st, offset: 0, index: 7)
                e.setBuffer(l.qkvBias ?? qkv, offset: 0, index: 8)
                e.setBuffer(ropeTab ?? qkv, offset: 0, index: 9)
                e.dispatchThreadgroups(MTLSize(width: qHeads + 16, height: B, depth: 1), threadsPerThreadgroup: tgHD)
                e.setComputePipelineState(try ctx.pipeline("dattn_nax" + kernelSuffix))
                for (i, b) in [q, l.kc, l.vt, opart, ml].enumerated() { e.setBuffer(b, offset: 0, index: i) }
                e.setBytes(&p, length: MemoryLayout<DAttnParams>.size, index: 5)
                e.setBuffer(st, offset: 0, index: 6)
                // 16 query rows (G heads × 16 / G block rows) per threadgroup.
                e.dispatchThreadgroups(
                    MTLSize(width: Int(p.nsplit), height: G * B / 16, depth: 8), threadsPerThreadgroup: tgHD)
                e.setComputePipelineState(try ctx.pipeline("dattn_reduce" + kernelSuffix))
                for (i, b) in [opart, ml, attn].enumerated() { e.setBuffer(b, offset: 0, index: i) }
                e.setBytes(&p, length: MemoryLayout<DAttnParams>.size, index: 3)
                e.dispatchThreadgroups(MTLSize(width: G * B, height: 8, depth: 1), threadsPerThreadgroup: tgHD)
            }
            if v1 {
                try S("o") { e in
                    try lin(e, x: attn, l.o, y: x, rows: B, accumulate: true)
                    if let ob = l.oBias { try addBias(e, ob) }
                }
            } else {
                try S("o") { try lin($0, x: attn, l.o, y: y, rows: B) }
                try S("conv") { try conv($0, finish: true, y, base: l.abase!, x) }
            }
            try S("norm") { try norm($0, x, l.n2, xn, rows: B) }
            if let mp = l.mproj, let mb = l.mbase {
                try S("convproj") { try lin($0, x: xn, mp, y: dyn, rows: B) }
                try S("conv") { try conv($0, finish: false, xn, base: mb, a) }
            }
            try S("gateup") { try lin($0, x: v1 ? xn : a, l.gu, y: gu, rows: B) }
            try S("swiglu") { e in
                var n = UInt32(inter)
                e.setComputePipelineState(try ctx.pipeline("dswiglu"))
                e.setBuffer(gu, offset: 0, index: 0)
                e.setBuffer(hb, offset: 0, index: 1)
                e.setBytes(&n, length: 4, index: 2)
                e.dispatchThreads(
                    MTLSize(width: B * inter, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            if v1 {
                try S("down") { try lin($0, x: hb, l.down, y: x, rows: B, accumulate: true) }
            } else {
                try S("down") { try lin($0, x: hb, l.down, y: y, rows: B) }
                try S("conv") { try conv($0, finish: true, y, base: l.mbase!, x) }
            }
        }
        try S("norm") { try norm($0, x, fnorm, hid, rows: B) }
        // 6-bit lm_head (Flash-Next): the same NAX path as the big model's head (FlashParts, nax6).
        // With qmv6m (7 rows) it cost 8.7 ms per round; it changes the order of the sums of the draft logits.
        let head = draftHead ?? lmHead
        if head.bits == 6 { vq.nax6 = true }
        try S("lm_head") { try vq.encode($0, x: hid, xOffset: H * 2, head, y: logits, rows: B - 1) }
        var V = UInt32(head.N)
        try S("topk") { e in
            e.setComputePipelineState(try ctx.pipeline("topk16_rows"))
            e.setBuffer(logits, offset: 0, index: 0)
            e.setBuffer(cidx, offset: 0, index: 1)
            e.setBuffer(cval, offset: 0, index: 2)
            e.setBytes(&V, length: 4, index: 3)
            e.dispatchThreadgroups(
                MTLSize(width: B - 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
            try encodeLogitsPost(e)
        }
        if v1 {
            // DFlash v1: the draft of each position is its best candidate (argmax), with its confidence in `conf`.
            try S("select") { e in
                var M = UInt32(B - 1)
                var tau = Self.confTau
                e.setComputePipelineState(try ctx.pipeline("dpick_top1"))
                e.setBuffer(cidx, offset: 0, index: 0)
                e.setBuffer(vtok, offset: 0, index: 1)
                e.setBytes(&M, length: 4, index: 2)
                e.setBuffer(cval, offset: 0, index: 3)
                e.setBuffer(conf, offset: 0, index: 4)
                e.setBytes(&tau, length: 4, index: 5)
                e.dispatchThreads(
                    MTLSize(width: B - 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: B - 1, height: 1, depth: 1))
            }
            return cur
        }
        try S("select") { e in
            try vq.encode(e, x: hid, xOffset: H * 2, hproj!, y: hp, rows: B - 1)
            if let tree {
                var bp = (UInt32(tree.nodes), UInt32(B - 1), UInt32(4), Float(1.5))
                e.setComputePipelineState(try ctx.pipeline("dtree_build"))
                for (i, b) in [cidx, cval, hp, pred!, succ!, vtok, tree.buf].enumerated() {
                    e.setBuffer(b, offset: 0, index: i)
                }
                e.setBytes(&bp, length: 16, index: 7)
                e.dispatchThreadgroups(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                return
            }
            var M = UInt32(B - 1)
            e.setComputePipelineState(try ctx.pipeline("dselect"))
            for (i, b) in [cidx, cval, hp, pred!, succ!, vtok].enumerated() { e.setBuffer(b, offset: 0, index: i) }
            e.setBytes(&M, length: 4, index: 6)
            e.setBuffer(conf, offset: 0, index: 7)
            var tau = Self.confTau
            e.setBytes(&tau, length: 4, index: 8)
            e.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        return cur
    }

    /// The o_proj bias (bf16 [H]) added to the block's fp32 rows `x`.
    private func addBias(_ e: MTLComputeCommandEncoder, _ bias: MTLBuffer) throws {
        var n = UInt32(H)
        e.setComputePipelineState(try ctx.pipeline("dbias_add"))
        e.setBuffer(x, offset: 0, index: 0)
        e.setBuffer(bias, offset: 0, index: 1)
        e.setBytes(&n, length: 4, index: 2)
        e.dispatchThreads(
            MTLSize(width: block * H, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// `logitsPost` on the top-16 values of the draft rows (`cval`), with the same kernel and order as the big model's
    /// logits; nothing without it.
    private func encodeLogitsPost(_ e: MTLComputeCommandEncoder) throws {
        guard let lp = logitsPost else { return }
        var pp = (UInt32((block - 1) * 16), lp.mul, Float(1), lp.cap)
        e.setComputePipelineState(try ctx.pipeline("logits_post"))
        e.setBuffer(cval, offset: 0, index: 0)
        e.setBytes(&pp, length: 16, index: 1)
        e.dispatchThreads(
            MTLSize(width: Int(pp.0), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    public func setBlockTokens(anchor: UInt32) {
        let p = dtok.contents().bindMemory(to: UInt32.self, capacity: block)
        p[0] = anchor
        for i in 1..<block { p[i] = maskToken }
    }
}
