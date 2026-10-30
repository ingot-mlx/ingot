import Foundation

/// Engine options. Set by whoever creates the model (the CLI: `lab serve` and the `lab-dev` benches) and carried in
/// `HybridModel.options`; the engine reads no environment variables except those in ref/api-surface.txt. They are read
/// each time they are used, so a bench can change them between one measurement and the next.
///
/// Defaults depend on the architecture (MoE or dense), the drafter and the command, not on the model name
/// (per-model enablement table):
/// - copy-from-context: on for MoE with a drafter and without one (LFM2.5), off for dense models with a drafter;
/// - verification rows from drafter confidence if the family has a calibration (`SpecRowsConfig.load`; today the Qwen3.6
///   MoE models, Flash-Next, and Gemma 4 26B-A4B and gpt-oss 20b with their v1 drafters) and 8 fixed otherwise (the
///   27B);
/// - GDN checkpoint inside the prefill pass in all that support it (not LFM2);
/// - 12-row draft tree on dense models with DFlash2, in greedy and with prompts of < 16K;
/// - exact prefill, except in `lab serve` with a MoE (`EngineOptions.serve(moe:)`: W8A8 from the family's position,
///   `ModelSpec.int8MinPos`, and Q·Kᵀ in int8 from 8K; per-family threshold).
public struct EngineOptions: Sendable, Equatable {
    /// Copy-from-context (CopySpec.swift).
    public enum CopyMode: String, Sendable, CaseIterable {
        /// Depends on the architecture (`copyEnabled`).
        case auto
        case on, off
    }

    public var copy: CopyMode = .auto
    /// Prefill in rotated W8A8 (lossy; --int8 1): dense GEMMs (GDN and attention projections, dense FFN);
    /// the MoE experts stay exact except Flash-Next's Q2 experts, with int8 activations (`ModelSpec.int8Experts`).
    public var prefillInt8 = false
    /// Prefill attention with Q·Kᵀ in int8 (lossy; --qk8 1) from `int8.qk8MinCtx`.
    public var prefillQK8 = false
    /// Settings for the two above.
    public var int8 = PrefillInt8Options()
    /// Rows of the draft tree on dense Qwen models with DFlash2, in greedy. 12 by default (it is exact and falls back to
    /// the chain on its own where it does not help); `--tree 0` goes back to the chain. With temperature, on
    /// MoE and with prompts of ≥ 16K the chain still applies. To be revisited with the per-Mac cost table.
    public var treeNodes = 12
    /// Bytes of the in-memory expert cache; the rest is read from the SSD (ExpertStream.swift). nil = all experts in
    /// memory (default), except on Flash-Next, whose experts always come from the SSD (8 GB cache). Qwen3.5/3.6 MoE and
    /// Flash-Next only; exact (same tokens).
    public var expertCacheBytes: Int?
    /// Read-ahead of each row's top-k with the next layer's router (0 = off; nil = 4, and 6
    /// on Flash-Next: 1 to 2% fewer ms per round with 14.43 GB than with 4; with 8, uneven).
    public var expertPrefetch: Int?
    /// Prefill chunks computed together layer by layer (each expert is fetched once per group; 1 = chunk by
    /// chunk). Each extra chunk takes chunkRows × hidden × 4 bytes (its residual stream).
    public var expertPrefillGroup = 4
    /// Flash-Next: chunks per prefill group (8 = a prompt of up to 16K in a single pass over the SSD experts;
    /// each extra chunk, 84 MB of residual stream with 2048-row chunks, only if the prompt uses them).
    public var flashPrefillGroup = 8
    /// With ≥ this many rows in the group, prefill fetches all missing experts of each layer ahead of the GPU
    /// (without stalling after the router: the read of layer l + 1 overlaps with the compute of l); with fewer, only
    /// those the router asks for, stalling at each layer.
    public var expertBulkMinRows = 256
    /// Flash-Next: the same in Flash-Next's grouped prefill. With a single chunk of fewer rows, only the experts it
    /// routes (the union of the top-k not in the cache, into the staging area), stalling at each layer; with more, all
    /// the missing ones once per group; with several chunks in the group, always all. With a warm expert cache, routing
    /// wins below ~900 rows (543 new tokens: 3.24 → 2.52 s, 20.2 → 11.4 GB; pi turns of 625–713 rows: −0.14 to
    /// −0.68 s), ties at 930–1000 and loses from ~1000 rows of code (1138: +0.31 s): the bulk read costs a flat
    /// ~3.2 s (SSD-bound) while routing grows with the rows. Both give the same bits.
    public var flashBulkMinRows = 900
    /// Flash-Next: wide read-ahead, top-K of each row into staging buffers that do not enter the cache unless
    /// the round uses them (ExpertWide.swift; 0 = read-ahead into the `expertPrefetch` cache). Exact. nil =
    /// 10 (−4 to −6% ms per round vs top-6 into the cache with the default `gen` cache;
    /// 8 and 12 similar, 6 and 16 worse). `expertPrefetch` > 0 is still required (the prediction).
    public var expertWide: Int?
    /// Flash-Next, lossy; nil = off; `lab serve` sets 5 for MoE: in rounds, experts missing from the cache whose best
    /// rank among the rows is ≥ N (0 = top-1) are not awaited: 5 = ranks 6–10, 8 = outside the top-8. Their weight
    /// becomes 0 (the rest of the row's experts are not renormalized: lower KL) and they are read in the background for
    /// later rounds.
    public var expertSkipRank: Int?
    /// Flash-Next: KV that grows on demand (KVPool: reserved for a span of 16K positions and reallocated as it grows).
    /// Set by the planner in `lab serve` (with `kvAllowance`, the positions the plan accounts for). false = the whole
    /// KV of the maximum context at load time.
    public var kvGrow = false
    public var kvAllowance = 65_536
    /// Flash-Next: bytes of the PLE table row cache on the CPU (FlashPLERows; splash uses 511 MiB).
    public var pleCacheBytes = 512_000_000
    /// Experimental options (`--exp key=value`, repeatable). Only for tests on branches: nothing reads them on main
    /// (`scripts/ci.sh` fails if `Sources/` reads `exp` outside this file: `scripts/api_surface.py --exp-check`).
    public var exp: [String: String] = [:]
    /// What only the benches and diagnostics touch.
    public var bench = BenchOptions()

    public init() {}

    /// What `lab serve` uses: for MoE, --int8 1 and --qk8 1 by default; dense models (27B,
    /// Bonsai) stay exact because they have not been measured. `expertSkipRank = 5` by default: only acts on Flash-Next
    /// with SSD experts (KL 5–9·10⁻⁴ in our use, pi +11%);
    /// `--expert-skip-rank 0` turns it off.
    public static func serve(moe: Bool) -> EngineOptions {
        var o = EngineOptions()
        o.prefillInt8 = moe
        o.prefillQK8 = moe
        o.expertSkipRank = moe ? 5 : nil
        return o
    }

    // periphery:ignore - main never calls it: the helper for the experiments of the branches, which remove their
    // `--exp` keys before delivery.
    /// Value of `--exp key=value` (`default` if missing or not parseable as T).
    public func exp<T: LosslessStringConvertible>(_ key: String, default d: T) -> T {
        exp[key].flatMap { T($0) } ?? d
    }

    /// Copy-from-context enabled? In `auto` (measured with pi against lab serve, same session): on for MoE with a
    /// drafter (Heretic −10.4% ms per decoded token on the agent task, Ornith −7.3%; prose −0.1 to −0.6%)
    /// and without one (LFM2.5 −13.8%); off for dense models with a drafter: on the 27B it does not win (−0.8%, within noise;
    /// only 4.4% of tokens copied) though it is exact.
    public func copyEnabled(moe: Bool, drafter: Bool) -> Bool {
        switch copy {
        case .on: return true
        case .off: return false
        case .auto: return !drafter || moe
        }
    }
}

/// Settings of the lossy prefill. Defaults are the chosen ones; the benches (`lab-dev kl --cfgs`) change them in code.
public struct PrefillInt8Options: Sendable, Equatable {
    /// Dense GEMMs that go in int8: gp/go = GDN projection and out_proj, ap/ao = attention q|k|v and o_proj (the 27B's
    /// dense FFN always).
    public var parts: Set<String> = ["gp", "go", "ap", "ao"]
    /// Layers that run in int8 (nil = all). Diagnostic.
    public var layers: Set<Int>?
    /// Position from which dense GEMMs run in int8 (earlier rows, exact; chunks are split there so that the
    /// state at each cut does not depend on how the prompt is split). nil = the family's
    /// (`ModelSpec.int8MinPos`); the benches set it (lab-dev kl, R= card).
    public var minPos: Int?
    /// Position from which Q·Kᵀ runs in int8 with `prefillQK8`.
    public var qk8MinCtx = 8192

    public init() {}
}

/// Options that only the benches and `lab-dev` diagnostics touch (they change nothing in `lab serve`).
public struct BenchOptions: Sendable, Equatable {
    /// Fixed rows per DFlash2 round (nil = the family's: by confidence if there is calibration, 8 if not).
    public var specRowsFixed: Int?
    /// Prefill GDN in chunks of 64 (WY form). false = sequential fp32 recurrence (exact, different rounding;
    /// noise reference of `lab-dev kl --cfgs noise`).
    public var gdnChunk = true
    /// Prints the ms of each `preparePrompt` step (passes, batches, final stretch, snapshots).
    public var prepTrace = false
    /// The copy policy does not measure times (it uses its priors and hit rates): rounds do not depend on machine
    /// load. Greedy and seeded runs already go untimed; this also covers unseeded sampling. Set by
    /// `gen --golden-logits`, whose fingerprint of the last round changed with the parallel lanes of `ci.sh --toy`
    /// even though the tokens were the same.
    public var copyPolicyUntimed = false

    public init() {}
}
