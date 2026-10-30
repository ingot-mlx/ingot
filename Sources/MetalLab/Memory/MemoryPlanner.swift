import Foundation

/// Values set by the user (nil = decided by the planner).
public struct MemoryOptions: Sendable, Equatable {
    public var maxContext: Int?
    public var prefixCacheBytes: Int?
    public var snapshots: Int?
    public var chunkRows: Int?
    public var drafter: Bool?
    /// SSD experts (Flash-Next): KV that grows on demand (KVPool; true by default): the plan counts the KV of
    /// `poolAllowance` positions and gives the native context if the whole KV fits by giving up the prefix cache, two
    /// snapshots and the slack. false: the whole context KV reserved at load (the previous behavior).
    public var kvPool = true
    public init(
        maxContext: Int? = nil, prefixCacheBytes: Int? = nil, snapshots: Int? = nil, chunkRows: Int? = nil,
        drafter: Bool? = nil
    ) {
        self.maxContext = maxContext
        self.prefixCacheBytes = prefixCacheBytes
        self.snapshots = snapshots
        self.chunkRows = chunkRows
        self.drafter = drafter
    }
}

/// Policy parameters (on 32 GB they reproduce the pre-planner values: 128K with Qwen3.6 and with the calibrated
/// 27B).
public struct MemoryPolicy: Sendable, Equatable {
    /// Reserve for macOS and the usual apps: max(3 GB, 20 % of physical).
    public var systemReserveMin = 3_000_000_000
    public var systemReserveFraction = 0.20
    /// Margin below `recommendedMaxWorkingSetSize`.
    public var gpuMargin = 1_000_000_000
    /// Slack for prefill peaks, command buffers and fragmentation.
    public var slack = 1_500_000_000
    /// Below this context the rest starts being trimmed.
    public var usefulContext = 32_768
    /// Below this context the model is not loaded.
    public var minimumContext = 8192
    public var defaultPrefixCache = 1_000_000_000
    public var defaultSnapshots = 4
    public var defaultChunk = 2048
    public init() {}
}

public struct MemoryPlan: Sendable {
    public var context: Int
    public var prefixCacheBytes: Int
    public var snapshots: Int
    public var chunkRows: Int
    public var drafter: Bool
    /// Budget B and expected total (with the prefix cache and snapshots full).
    public var budget: Int
    public var total: Int
    /// Expected `currentAllocatedSize` right after load (no prefix cache or snapshots).
    public var expectedAtLoad: Int
    public var breakdown: [(String, Int)]
    /// SSD experts (Flash-Next): bytes of expert cache in memory (what is left of the budget).
    public var expertCacheBytes: Int?
    /// KV that grows on demand (KVPool) and the positions the plan counts.
    public var kvGrow = false
    public var kvAllowance = 0
    /// What was decided and why, in order.
    public var explanation: [String]
    /// Warnings (user options that do not fit, estimated calibration, memory used by other apps...).
    public var warnings: [String]
    public var fits: Bool { total <= budget }
}

public enum MemoryPlanError: Error, CustomStringConvertible {
    case doesNotFit(String)
    public var description: String {
        switch self {
        case .doesNotFit(let s): return s
        }
    }
}

/// Memory planner: a pure function of the machine, the model footprint and the user options.
///
/// Budget: `B = min(rws - margin, physical - reserve) - slack`. Total: weights + drafter + prefill work +
/// snapshots + prefix cache + KV x context. Context is the native one, or half, a quarter..., the largest that
/// fits with everything else at its default. If that falls below 32K, things are trimmed, least noticeable first:
/// prefix cache (1 -> 0.25 GB -> off), snapshots (4 -> 2), prefill chunk (2048 -> 1024), context down to 16K,
/// drafter off and context down to 8K. If it still does not fit, it is not loaded. It never changes model.
/// What the user sets wins: the rest is adjusted and, if it does not fit, a warning is given.
public enum MemoryPlanner {
    /// `cpuBytes`: CPU memory of the model (the PLE row cache of Flash-Next). It is not Metal memory, so it counts
    /// against physical, not rws: it only trims the budget if physical is the binding limit.
    public static func budget(_ m: MachineInfo, cpuBytes: Int = 0, policy p: MemoryPolicy = MemoryPolicy()) -> Int {
        let reserve = max(p.systemReserveMin, Int(Double(m.physicalMemory) * p.systemReserveFraction))
        return min(m.recommendedWorkingSet - p.gpuMargin, m.physicalMemory - reserve - cpuBytes) - p.slack
    }

    struct Level {
        var prefixCache: Int, snapshots: Int, chunk: Int, drafter: Bool, minContext: Int, why: String
    }

    /// Trim steps, in order.
    static func levels(_ p: MemoryPolicy, drafter: Bool) -> [Level] {
        let u = p.usefulContext
        var l = [
            Level(
                prefixCache: p.defaultPrefixCache, snapshots: p.defaultSnapshots, chunk: p.defaultChunk,
                drafter: drafter, minContext: u,
                why: "default values"),
            Level(
                prefixCache: 250_000_000, snapshots: p.defaultSnapshots, chunk: p.defaultChunk, drafter: drafter,
                minContext: u,
                why: "prefix cache at 0.25 GB"),
            Level(
                prefixCache: 0, snapshots: p.defaultSnapshots, chunk: p.defaultChunk, drafter: drafter, minContext: u,
                why: "prefix cache off"),
            Level(
                prefixCache: 0, snapshots: 2, chunk: p.defaultChunk, drafter: drafter, minContext: u,
                why: "snapshots 4 -> 2"),
            Level(
                prefixCache: 0, snapshots: 2, chunk: 1024, drafter: drafter, minContext: u,
                why: "prefill chunks 2048 -> 1024"),
            Level(
                prefixCache: 0, snapshots: 2, chunk: 1024, drafter: drafter, minContext: 16_384,
                why: "context down to 16K"),
        ]
        if drafter {
            l.append(
                Level(
                    prefixCache: 0, snapshots: 2, chunk: 1024, drafter: false, minContext: 16_384,
                    why: "no drafter (writing ~40 % slower)"))
        }
        l.append(
            Level(
                prefixCache: 0, snapshots: 2, chunk: 1024, drafter: false, minContext: p.minimumContext,
                why: "context down to 8K"))
        return l
    }

    struct Choice {
        var context: Int, prefixCache: Int, snapshots: Int, chunk: Int, drafter: Bool
        /// KVPool: the KV counts `poolAllowance` positions (anything beyond is covered by the prefix cache, two
        /// snapshots and the slack).
        var pool = false
    }

    /// KV positions the plan counts with KVPool (like splash's resident allocation).
    public static let poolAllowance = 65_536

    static func parts(_ f: ModelFootprint, _ c: Choice) -> [(String, Int)] {
        var r: [(String, Int)] = [("weights", f.weightBytes)]
        if c.drafter, let d = f.drafterBytes { r.append(("drafter", d)) }
        r.append(("fixed state and prefill work (chunks of \(c.chunk))", f.prefillBytes(chunkRows: c.chunk)))
        r.append(("snapshots (\(c.snapshots))", c.snapshots * f.snapshotBytes(drafter: c.drafter)))
        r.append(("prefix cache", c.prefixCache))
        if c.pool {
            let a = min(c.context, poolAllowance)
            r.append(
                ("KV cache (grows on demand up to \(c.context) tokens; the plan counts \(a))", f.kvBytesPerToken * a))
        } else {
            r.append(("KV cache (\(c.context) tokens)", f.kvBytesPerToken * c.context))
            if f.windowLayers > 0 {
                r.append(
                    ("KV of the \(f.windowLayers) sliding-window layers (ring)", f.kvFixedBytes(chunkRows: c.chunk)))
            }
        }
        if let st = f.streamed {
            let g = st.groupBytes(chunkRows: c.chunk, drafter: c.drafter)
            if g > 0 { r.append(("extra chunks of the prefill group (\(st.prefillGroup) per group)", g)) }
            if st.cpuCacheBytes > 0 {
                r.append(("PLE row cache (CPU: counts against physical, not Metal)", st.cpuCacheBytes))
            }
            r.append(("expert cache (minimum; takes what is left)", st.fixedCacheBytes ?? st.minCacheBytes))
        }
        return r
    }

    /// Expected Metal memory: everything but the PLE row cache, which is CPU memory (counted in `budget`).
    static func total(_ f: ModelFootprint, _ c: Choice) -> Int {
        parts(f, c).reduce(0) { $0 + $1.1 } - (f.streamed?.cpuCacheBytes ?? 0)
    }

    /// Candidate contexts: the native one and its halves down to `minContext` (like the old `defaultMaxCtx`).
    static func contexts(native: Int, down to: Int) -> [Int] {
        var r: [Int] = []
        var n = native
        repeat { r.append(n); n /= 2 } while n >= min(to, native) && n > 0
        return r.filter { $0 >= min(to, native) }
    }

    /// Context the plan counts when it moves the experts to the SSD itself (`ModelFootprint.streamable`).
    public static let streamableContext = 65_536

    /// The plan, or `MemoryPlanError.doesNotFit` if the model does not fit even at the minimum.
    ///
    /// A model whose experts can be read from the SSD (`streamable`) stays in memory if it fits there with the
    /// useful context (or the one set) and its drafter. Otherwise its experts go to the SSD and the expert cache
    /// takes what is left: before a prefix cache over 0.25 GB and a context over 64K (every 0.5 GB of cache is
    /// 12–15 % more tok/s on a 16 GB Mac), then the usual trims with the minimum cache.
    public static func plan(
        machine m: MachineInfo, model f: ModelFootprint, options o: MemoryOptions = MemoryOptions(),
        policy p: MemoryPolicy = MemoryPolicy()
    ) throws -> MemoryPlan {
        guard let st = f.streamable, f.streamed == nil else {
            return try planOne(machine: m, model: f, options: o, policy: p)
        }
        var inMemory = f
        inMemory.streamable = nil
        let wantsDrafter = f.hasDrafter && o.drafter != false
        let mem = try? planOne(machine: m, model: inMemory, options: o, policy: p)
        if let mp = mem, mp.fits, mp.drafter || !wantsDrafter,
            o.maxContext != nil || mp.context >= min(p.usefulContext, f.nativeContext)
        {
            return mp
        }
        var ssd = inMemory
        ssd.weightBytes -= st.totalBytes
        ssd.streamed = st
        ssd.streamed!.automatic = true
        var plan = try planOne(machine: m, model: ssd, options: o, policy: p)
        let gb = { (b: Int) in String(format: "%.2f GB", Double(b) / 1e9) }
        let why =
            mem.map {
                "in memory it would need \($0.context) tokens\($0.drafter ? "" : ", no drafter")"
                    + "\($0.fits ? "" : " and does not fit")"
            } ?? "it does not fit in memory"
        plan.explanation.insert(
            "\(why): the \(gb(st.totalBytes)) of experts are read from the SSD and the expert cache takes what is left",
            at: m.simulated == nil ? 1 : 2)
        return plan
    }

    static func planOne(
        machine m: MachineInfo, model f: ModelFootprint, options o: MemoryOptions, policy p: MemoryPolicy
    ) throws -> MemoryPlan {
        let B = budget(m, cpuBytes: f.streamed?.cpuCacheBytes ?? 0, policy: p)
        let gb = { (b: Int) in String(format: "%.2f GB", Double(b) / 1e9) }
        var warnings: [String] = []
        if f.largestTensorBytes > m.maxBufferLength {
            throw MemoryPlanError.doesNotFit(
                "\(f.name): its largest tensor (\(gb(f.largestTensorBytes))) does not fit in a Metal buffer on this Mac (max \(gb(m.maxBufferLength)))."
            )
        }
        if o.drafter == true && !f.hasDrafter {
            warnings.append("drafter requested but \(f.name) has none: running without drafter")
        }
        let canDraft = f.hasDrafter && o.drafter != false
        // The longest context the max buffer allows for one layer's KV.
        let maxCtxBuffer = m.maxBufferLength / max(1, f.kvBytesPerTokenPerLayer)
        func apply(_ l: Level) -> Choice {
            Choice(
                context: 0, prefixCache: o.prefixCacheBytes ?? l.prefixCache, snapshots: o.snapshots ?? l.snapshots,
                chunk: o.chunkRows ?? l.chunk, drafter: o.drafter.map { $0 && f.hasDrafter } ?? (canDraft && l.drafter))
        }
        let reserve = max(p.systemReserveMin, Int(Double(m.physicalMemory) * p.systemReserveFraction))
        var explanation = [
            "budget \(gb(B)) = min(rws \(gb(m.recommendedWorkingSet)) - \(gb(p.gpuMargin)), physical \(gb(m.physicalMemory)) - reserve \(gb(reserve)))"
                + " - slack \(gb(p.slack))"
        ]
        var chosen: Choice?
        var tried: [String] = []
        // SSD experts: each GB of KV or prefix cache is one GB less of expert cache, so the default context is
        // capped (128K with >= 20 GB of budget, 64K with less; splash keeps ~147K in memory). With KVPool (default),
        // context is the native one if the whole KV fits by giving up the prefix cache, snapshots beyond two and the
        // slack (what InferenceEngine does when the KV goes past the allocation).
        // Experts the planner moved to the SSD itself: 64K counted and the trims from a 0.25 GB prefix cache on.
        let auto = f.streamed?.automatic == true
        let pool = f.streamed.map { $0.onDemandKV && o.kvPool && $0.fixedCacheBytes == nil } ?? false
        let native =
            f.streamed == nil || pool
            ? f.nativeContext
            : min(f.nativeContext, auto ? streamableContext : B >= 20_000_000_000 ? 131_072 : 65_536)
        for l in levels(p, drafter: canDraft).dropFirst(auto ? 1 : 0) {
            var c = apply(l)
            c.pool = pool
            func fullFits(_ t: Choice) -> Bool {
                guard t.pool else { return true }
                let extra = f.kvBytesPerToken * (t.context - min(t.context, poolAllowance))
                return extra <= t.prefixCache + max(0, t.snapshots - 2) * f.snapshotBytes(drafter: t.drafter) + p.slack
            }
            if let fixed = o.maxContext {
                c.context = fixed
                if total(f, c) <= B && fullFits(c) && fixed <= maxCtxBuffer {
                    chosen = c; explanation.append("context \(fixed) (set); \(l.why)"); break
                }
            } else if let n = contexts(native: native, down: l.minContext).first(where: { ctx in
                var t = c
                t.context = ctx
                return ctx <= maxCtxBuffer && total(f, t) <= B && fullFits(t)
            }) {
                c.context = n
                chosen = c
                explanation.append(
                    "context \(n) (native \(f.nativeContext)\(native < f.nativeContext ? ", capped at \(native) by the SSD experts" : "")); \(l.why)"
                )
                break
            }
            tried.append(l.why)
        }
        if !tried.isEmpty && chosen != nil {
            explanation.insert("did not fit with: " + tried.joined(separator: "; ") + " -> trimmed", at: 1)
        }
        if chosen == nil {
            // Is it due to what the user set? The minimum without their values.
            let last = levels(p, drafter: false).last!
            var minimal = Choice(
                context: min(p.minimumContext, f.nativeContext), prefixCache: last.prefixCache,
                snapshots: last.snapshots,
                chunk: last.chunk, drafter: false)
            let minimalTotal = total(f, minimal)
            if minimalTotal > B {
                let need = minimalTotal
                throw MemoryPlanError.doesNotFit(
                    "\(f.name) does not fit on this Mac\(m.simulated.map { " (\($0))" } ?? ""): even with a context of \(minimal.context) tokens, no drafter, "
                        + "no prefix cache, 2 snapshots and chunks of 1024 it needs \(gb(need)) (weights \(gb(f.weightBytes))) and the Mac leaves \(gb(B)) "
                        + "(\(gb(m.recommendedWorkingSet)) for the GPU out of \(gb(m.physicalMemory)) of memory). A smaller model is needed "
                        + "(fewer parameters or fewer bits: about \(gb(max(0, f.weightBytes - (need - B)))) of weights at most)."
                )
            }
            // The minimum fits, but not with what the user set: theirs wins, with a warning.
            var c = apply(last)
            c.context = o.maxContext ?? minimal.context
            minimal = c
            chosen = c
            warnings.append(
                "what you set does not fit: the plan needs \(gb(total(f, c))) and the budget is \(gb(B)); "
                    + "used anyway (may be slow or fail to load)")
            explanation.append("with the set values it fits at no step; the rest at minimum")
        }
        let c = chosen!
        if o.maxContext == nil, c.context < native {
            var next = c
            next.context = c.context * 2
            explanation.append("double (\(next.context)) would need \(gb(total(f, next))) of \(gb(B))")
        }
        if canDraft && !c.drafter { warnings.append("no drafter so that it fits: writing will be ~40 % slower") }
        if !f.calibration.measured {
            warnings.append("calibration of \(f.modelType) estimated, not measured (\(f.calibration.source))")
        }
        var t = total(f, c)
        var parts = parts(f, c)
        var expertCache: Int?
        if let st = f.streamed, c.pool {
            let cache = min(st.totalBytes, st.minCacheBytes + max(0, B - t))
            expertCache = cache
            t += cache - st.minCacheBytes
            let kvFull = f.kvBytesPerToken * c.context, kvA = f.kvBytesPerToken * min(c.context, poolAllowance)
            parts[parts.count - 1] = (
                String(
                    format: "expert cache (%.0f %% of %.1f GB; the rest on the SSD)",
                    100 * Double(cache) / Double(st.totalBytes), Double(st.totalBytes) / 1e9), cache
            )
            explanation.append(
                String(
                    format:
                        "on-demand KV: with %d tokens it takes %.2f GB more than counted; "
                        + "the prefix cache, snapshots beyond 2 and the slack cover it",
                    c.context, Double(kvFull - kvA) / 1e9))
        } else if let st = f.streamed {
            let cache = st.fixedCacheBytes ?? min(st.totalBytes, st.minCacheBytes + max(0, B - t))
            expertCache = cache
            t += cache - (st.fixedCacheBytes ?? st.minCacheBytes)
            parts[parts.count - 1] = (
                String(
                    format: "expert cache (%.0f %% of %.1f GB; the rest on the SSD)",
                    100 * Double(cache) / Double(st.totalBytes), Double(st.totalBytes) / 1e9), cache
            )
        }
        if let avail = m.availableMemory, m.simulated == nil, t > avail + f.weightBytes {
            // Weights are usually already in the file cache (counted as inactive or in use): warn only if not even so.
            warnings.append(
                "\(gb(avail)) free or reclaimable now; with other apps open macOS will have to compress memory"
            )
        }
        if let s = m.simulated { explanation.insert(s, at: 0) }
        return MemoryPlan(
            context: c.context, prefixCacheBytes: c.prefixCache, snapshots: c.snapshots, chunkRows: c.chunk,
            drafter: c.drafter,
            budget: B, total: t, expectedAtLoad: t - c.prefixCache - c.snapshots * f.snapshotBytes(drafter: c.drafter),
            breakdown: parts, expertCacheBytes: expertCache, kvGrow: c.pool,
            kvAllowance: c.pool ? min(c.context, poolAllowance) : 0, explanation: explanation, warnings: warnings)
    }
}

extension MemoryPlanner {
    /// lab serve batching: conversations per round (up to `maxSlots`), positions and snapshots of each extra
    /// slot that fit in what the plan leaves of the budget. Each extra slot takes its KV (and the ring of the window
    /// layers), its recurrent state and its own snapshots; with a drafter (slots served one at a time), also
    /// its drafter context ring, which its snapshots carry too (one more ring is in flight while binding). Most slots
    /// first, each with the longest context (the plan's capped at `maxPositions`, its half... down to 16K) and, before
    /// a shorter context, the plan's snapshots or else 2 (the two points of a prompt; the client keeps the context it
    /// sees), and the bytes they take; (1, 0, 0, 0) if not even two fit. Slot 0 keeps the plan's whole
    /// context and snapshots; a request that does not fit an extra slot waits for it.
    public static func batchSlots(
        plan: MemoryPlan, model f: ModelFootprint, maxSlots: Int = 4, maxPositions: Int = 32_768
    ) -> (
        slots: Int, positions: Int, snapshots: Int, bytes: Int
    ) {
        let ring = plan.drafter ? f.snapshotDrafterBytes : 0
        let left = plan.budget - plan.total - ring
        // Its snapshots and its live state (one more), its KV and its drafter ring.
        func slotBytes(_ pos: Int, _ snaps: Int) -> Int {
            f.kvBytesPerToken * pos + (f.windowLayers > 0 ? f.kvFixedBytes(chunkRows: plan.chunkRows) : 0)
                + (snaps + 1) * f.snapshotBytes(drafter: plan.drafter) + ring
        }
        let snapChoices = plan.snapshots > 2 ? [plan.snapshots, 2] : [plan.snapshots]
        for n in stride(from: maxSlots, through: 2, by: -1) {
            for pos in contexts(native: min(plan.context, maxPositions), down: 16_384) {
                if let k = snapChoices.first(where: { (n - 1) * slotBytes(pos, $0) <= left }) {
                    return (n, pos, k, (n - 1) * slotBytes(pos, k) + ring)
                }
            }
        }
        return (1, 0, 0, 0)
    }
}

extension MemoryPlan {
    /// Multi-line summary for the console.
    public func describe() -> String {
        let gb = { (b: Int) in String(format: "%.2f GB", Double(b) / 1e9) }
        var s =
            "memory plan: context \(context), prefix cache \(gb(prefixCacheBytes)), \(snapshots) snapshots, "
            + "chunks of \(chunkRows), \(drafter ? "with" : "no") drafter; total \(gb(total)) of \(gb(budget))\n"
        for (k, v) in breakdown { s += "  \(k): \(gb(v))\n" }
        for e in explanation { s += "  · \(e)\n" }
        for w in warnings { s += "  warning: \(w)\n" }
        return s
    }
}
