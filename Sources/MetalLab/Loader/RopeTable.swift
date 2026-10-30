import Foundation

// Portions adapted from mlx-lm (MIT, Copyright (c) 2023 Apple Inc.; `rope_utils.py`; see THIRD_PARTY_NOTICES.md).
// RoPE variants that only change the per-dimension frequencies (and a magnitude scale), computed on the CPU
// the way mlx-lm's `initialize_rope` does (rope_utils.py): `linear` and `yarn`. The kernels read the table
// (`inv_freq` per rotated pair plus `mscale` at the end) only in models that have a scaled layer (LAB_AT_ROPETAB); the
// default RoPE keeps its `pow(theta, …)` on the GPU, so models without scaling run exactly as before.
// Out, with a load error: `dynamic` (depends on the length), `longrope` (two tables), `mscale_all_dim` (mlx-lm folds
// it into the attention scale) and `llama3` (only Llama, tier 2; removed). `proportional` (Gemma 4's full
// layers) is a table over the whole head with zero frequencies past its partial factor.

/// Frequency scaling of a RoPE (`rope_scaling` or `rope_parameters` with a `rope_type` other than default).
public struct RopeScaling: Sendable, Equatable {
    public enum Kind: String, Sendable { case linear, yarn, proportional }
    public var kind: Kind
    /// `factor` (every kind).
    public var factor: Double
    /// `original_max_position_embeddings` (yarn).
    public var originalMax: Double = 0
    /// YaRN: `beta_fast`, `beta_slow`, `mscale` and `truncate` (transformers: false in gpt-oss, the correction range
    /// is not rounded to whole dimensions; mlx-lm always rounds it); `attentionFactor` = transformers'
    /// `attention_factor` (overrides the mscale computed from `factor`).
    public var betaFast: Double = 32, betaSlow: Double = 1, mscale: Double = 1
    public var truncate = true
    public var attentionFactor: Double?
    /// proportional: `partial_rotary_factor` (the first `partial · dims / 2` pairs rotate, the rest keep angle 0).
    public var partial: Double = 1

    public init(kind: Kind, factor: Double) { self.kind = kind; self.factor = factor }

    /// The RoPE type of a parameter dict (`rope_type`, or the older `type`; default if neither).
    public static func type(_ p: [String: Any]) -> String {
        (p["type"] as? String) ?? (p["rope_type"] as? String) ?? "default"
    }

    /// What the engine cannot run in a RoPE parameter dict (nil if it can).
    public static func gap(_ p: [String: Any]) -> String? {
        do {
            _ = try parse(p)
            return nil
        } catch SafetensorsError.format(let m) {
            return m
        } catch {
            return "\(error)"
        }
    }

    /// Scaling of a RoPE parameter dict: nil for the default RoPE; throws with what the engine cannot run.
    public static func parse(_ p: [String: Any]) throws -> RopeScaling? {
        func num(_ k: String) -> Double? { (p[k] as? NSNumber)?.doubleValue }
        let ty = type(p)
        switch ty {
        case "default": return nil
        case "proportional":
            // mlx-lm ProportionalRoPE: periods factor · base^(2i/dims) for the rotated pairs, infinite for the rest.
            var s = RopeScaling(kind: .proportional, factor: num("factor") ?? 1)
            s.partial = num("partial_rotary_factor") ?? 1
            guard s.factor > 0, s.partial > 0, s.partial <= 1 else {
                throw SafetensorsError.format("RoPE «proportional» with factor \(s.factor), partial \(s.partial)")
            }
            return s
        case "linear", "yarn":
            guard let kind = Kind(rawValue: ty), let factor = num("factor"), factor > 0 else {
                throw SafetensorsError.format("RoPE «\(ty)» without a positive factor")
            }
            var s = RopeScaling(kind: kind, factor: factor)
            if ty == "linear" { return s }
            guard let orig = num("original_max_position_embeddings"), orig > 0 else {
                throw SafetensorsError.format("RoPE «\(ty)» without original_max_position_embeddings")
            }
            s.originalMax = orig
            if let m = num("mscale_all_dim"), m != 0 {
                throw SafetensorsError.format("RoPE yarn with mscale_all_dim (folded into the attention scale)")
            }
            s.betaFast = num("beta_fast") ?? 32
            s.betaSlow = num("beta_slow") ?? 1
            s.mscale = num("mscale") ?? 1
            s.truncate = (p["truncate"] as? Bool) ?? true
            s.attentionFactor = num("attention_factor")
            return s
        default:
            throw SafetensorsError.format(
                "RoPE with rope_type «\(ty)» (supported: default, linear, yarn, proportional)")
        }
    }
}

/// Per-layer RoPE table for the attention kernels: `inv_freq[i]` for the `rotaryDims / 2` rotated pairs (angle =
/// position · inv_freq[i]) and, in the last entry, `mscale`, which multiplies the rotated dimensions of q and k.
public enum RopeTable {
    /// Table of a RoPE: computed in Double and rounded to Float once. Without scaling, the default RoPE
    /// (`theta^(-i / (rotaryDims/2))`, the formula of the kernels).
    public static func make(theta: Double, rotaryDims dims: Int, scaling: RopeScaling?) -> [Float] {
        let rh = dims / 2
        // Period of each pair (mlx-lm's `freqs`): base^(2i / dims).
        let period = (0..<rh).map { pow(theta, Double(2 * $0) / Double(dims)) }
        var inv = period.map { 1 / $0 }
        var mscale = 1.0
        switch scaling {
        case nil: break
        case let s? where s.kind == .linear: inv = inv.map { $0 / s.factor }
        case let s? where s.kind == .proportional:
            // mlx-lm ProportionalRoPE over the whole head: `int(dims · partial)` rotated dimensions, the period of
            // pair i factor · base^(2i/dims); the other pairs have an infinite period (angle 0: cos 1, sin 0, exact).
            let rotated = Int(Double(dims) * s.partial) / 2
            inv = (0..<rh).map { i in i < rotated ? 1 / (s.factor * period[i]) : 0 }
        case let s?:
            let (low, high) = yarnRange(theta: theta, dims: dims, s)
            inv = (0..<rh).map { i in
                // Extrapolation weight (mlx-lm `freq_mask`): 1 below `low` (original frequency), 0 above `high`
                // (interpolated: frequency / factor), linear in between.
                let ramp = min(max((Double(i) - low) / (high - low), 0), 1)
                let extra = 1 - ramp
                return inv[i] * extra + inv[i] / s.factor * (1 - extra)
            }
            mscale = s.attentionFactor ?? yarnMscale(s.factor, s.mscale)
        }
        return inv.map { Float($0) } + [Float(mscale)]
    }

    /// YaRN correction range in pairs (mlx-lm `yarn_find_correction_range`; transformers with `truncate`), with
    /// the 0.001 nudge of mlx-lm when both ends meet.
    static func yarnRange(theta: Double, dims: Int, _ s: RopeScaling) -> (Double, Double) {
        func dim(_ rotations: Double) -> Double {
            Double(dims) * log(s.originalMax / (rotations * 2 * Double.pi)) / (2 * log(theta))
        }
        var low = dim(s.betaFast), high = dim(s.betaSlow)
        if s.truncate { low = low.rounded(.down); high = high.rounded(.up) }
        low = max(low, 0); high = min(high, Double(dims - 1))
        if low == high { high += 0.001 }
        return (low, high)
    }

    /// `yarn_get_mscale`: 0.1 · mscale · ln(factor) + 1 (1 without scaling).
    static func yarnMscale(_ factor: Double, _ mscale: Double) -> Double {
        factor <= 1 ? 1 : 0.1 * mscale * log(factor) + 1
    }
}
