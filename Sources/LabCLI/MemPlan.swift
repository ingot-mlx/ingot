import Foundation
import Metal
import MetalLab

// Startup according to the Mac's memory: joins the planner (MetalLab/Memory) with the CLI. Options shared by serve,
// gen and plan:
//   --mem-limit-gb N   simulates a Mac with N GB (16, 24, 48…) on this one; --rws-gb X also sets its GPU memory
//   --max-ctx N, --prefix-cache-gb X, --snapshots N, --chunk N, --drafter none   set that value (they override the plan)
//   --kv-pool 0|1      (Flash-Next) KV that grows on demand: the plan counts 64K and gives the native context (1)

/// Values set by the user.
package func memOptions() -> MemoryOptions {
    var o = MemoryOptions()
    o.maxContext = args.opts["max-ctx"].flatMap { Int($0) }
    o.prefixCacheBytes = args.opts["prefix-cache-gb"].flatMap { Double($0) }.map { Int($0 * 1e9) }
    o.snapshots = args.opts["snapshots"].flatMap { Int($0) }
    o.chunkRows = args.opts["chunk"].flatMap { Int($0) }
    if args.opts["drafter"] == "none" { o.drafter = false }
    // Flash-Next: --kv-pool 0 returns to the full-context KV at load (the previous behavior).
    if let v = args.opts["kv-pool"] { o.kvPool = v == "1" || v == "on" }
    return o
}

/// The real machine or the one simulated with --mem-limit-gb.
package func planMachine(device: MTLDevice?) -> MachineInfo {
    let real = MachineInfo.current(device: device)
    guard let n = args.opts["mem-limit-gb"].flatMap({ Double($0) }) else { return real }
    return real.simulating(memoryGB: n, rwsGB: args.opts["rws-gb"].flatMap { Double($0) })
}

/// The drafter the memory plan counts (and lab serve, lab run and lab info use): resolvedDrafterDir with the toy's
/// `drafter/`, or nil if the drafter cannot be found (the plan goes without it). A drafter that does not fit
/// (--drafter or the pack's) is an error.
package func plannedDrafterDir(_ dir: URL, _ cfg: ModelConfig) throws -> URL? {
    do { return try resolvedDrafterDir(dir, cfg, toyPack: true) } catch SafetensorsError.io(_) { return nil }
}

/// Plan for `dir` on this machine (or the simulated one). Prints it and throws `MemoryPlanError` if it does not fit.
/// `loading` (lab serve, lab run): a default drafter missing from the cache is a load error that says how to get it
/// (without --drafter none), not a plan without drafter; lab plan and lab-dev go on without it.
package func memoryPlan(
    dir: URL, cfg: ModelConfig, device: MTLDevice?, options: MemoryOptions, loading: Bool = false
) throws -> MemoryPlan {
    let dd = loading ? try resolvedDrafterDir(dir, cfg, toyPack: true) : try plannedDrafterDir(dir, cfg)
    let fp = try plannedFootprint(dir: dir, cfg: cfg, drafterDir: dd)
    let m = planMachine(device: device)
    let plan = try MemoryPlanner.plan(machine: m, model: fp, options: options)
    print(plan.describe(), terminator: "")
    return plan
}

/// The model's footprint; with --expert-cache-gb X (SSD expert streaming), weights count X GB of experts instead of all (the rest
/// stays on the SSD). Without it, a MoE whose experts can be streamed (`streamable`) is moved to the SSD by the plan
/// only if it does not fit in memory.
package func plannedFootprint(dir: URL, cfg: ModelConfig, drafterDir: URL?) throws -> ModelFootprint {
    var fp = try ModelFootprint(directory: dir, drafterDirectory: drafterDir)
    if args.opts["expert-cache-gb"] != nil {
        fp.streamable = nil
    } else if fp.streamable != nil, let g = args.opts["expert-pf-group"].flatMap({ Int($0) }) {
        fp.streamable = ModelFootprint.streamableExperts(
            cfg, tensors: try ModelFootprint.headerTensors(directory: dir), prefillGroup: g)
    }
    if fp.streamed != nil {
        // Flash-Next: experts always from the SSD; --expert-cache-gb sets its cache, --ple-cache-mb the PLE one.
        fp.streamed!.fixedCacheBytes = args.opts["expert-cache-gb"].flatMap { Double($0) }.map { Int($0 * 1e9) }
        if let mb = args.opts["ple-cache-mb"].flatMap({ Int($0) }) { fp.streamed!.cpuCacheBytes = mb * 1_000_000 }
        fp.streamed!.prefillGroup =
            args.opts["expert-pf-group"].flatMap { Int($0) } ?? EngineOptions().flashPrefillGroup
        return fp
    }
    if let g = args.opts["expert-cache-gb"].flatMap({ Double($0) }), cfg.isMoE {
        let experts = try ModelFootprint.headerTensors(directory: dir)
            .filter { $0.name.contains("switch_mlp") && !$0.name.contains("mtp") }.reduce(0) { $0 + $1.bytes }
        let cache = min(Int(g * 1e9), experts)
        // Grouped prefill carries each chunk's residual stream extra (fp32) and the drafter captures of
        // a second chunk (half, 5 layers); counted with chunks of 2048 (the maximum).
        let groups = max(1, args.int("expert-pf-group", 4))
        let H = cfg.hiddenSize
        let groupBytes = groups > 1 ? 2048 * ((groups - 1) * H * 4 + 5 * H * 2) : 0
        fp.weightBytes += cache - experts + groupBytes
        print(
            String(
                format: "SSD streaming: %.2f GB of experts in memory of %.2f GB (rest from SSD)", Double(cache) / 1e9,
                Double(experts) / 1e9))
    }
    return fp
}

/// After loading: if Metal memory exceeds the forecast by more than 5%, the family's calibration is wrong.
package func checkPlanAfterLoad(_ plan: MemoryPlan, device: MTLDevice) {
    let now = device.currentAllocatedSize
    if Double(now) > Double(plan.expectedAtLoad) * 1.05 {
        print(
            String(
                format:
                    "warning: Metal memory %.2f GB after load, %.0f%% above the forecast (%.2f GB): check the family calibration",
                Double(now) / 1e9, (Double(now) / Double(plan.expectedAtLoad) - 1) * 100,
                Double(plan.expectedAtLoad) / 1e9))
    }
}

/// lab-dev gen: the prompts set the context, the mode sets the drafter, and there is no prefix cache or snapshots; the plan
/// checks that it fits and picks the prefill chunk (or the --chunk one).
package func genMemOptions(maxCtx: Int, drafter: Bool) -> MemoryOptions {
    var o = memOptions()
    o.maxContext = maxCtx
    o.prefixCacheBytes = 0
    o.snapshots = 0
    o.drafter = drafter
    // lab-dev gen keeps the full KV at load unless --kv-pool 1.
    if args.opts["kv-pool"] == nil { o.kvPool = false }
    return o
}

/// SSD experts (Flash-Next, or a MoE the plan moved to the SSD): the plan's expert cache, if --expert-cache-gb does not
/// set it.
package func applyPlan(_ o: inout EngineOptions, _ plan: MemoryPlan) {
    if plan.kvGrow {
        o.kvGrow = true
        o.kvAllowance = plan.kvAllowance
    }
    if o.expertCacheBytes == nil, let x = plan.expertCacheBytes { o.expertCacheBytes = x }
}

/// lab serve batching: conversations per round and positions of each extra slot (MemoryPlanner.batchSlots).
/// `drafterDir`: the drafter lab serve opened (its ring goes in each slot and its snapshots).
package func batchPlan(
    dir: URL, cfg: ModelConfig, plan: MemoryPlan, drafterDir: URL?
) throws -> (slots: Int, positions: Int, snapshots: Int, bytes: Int) {
    let fp = try plannedFootprint(dir: dir, cfg: cfg, drafterDir: drafterDir)
    return MemoryPlanner.batchSlots(plan: plan, model: fp)
}
