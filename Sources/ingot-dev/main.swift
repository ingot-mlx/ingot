import Foundation
import Ingot
import IngotCLI

// ingot-dev: accuracy tests and benches of ingot (ingot, to use the engine: serve, run, plan, info). Details of each
// command in docs/development.md.

func unknown(_ what: String, _ value: String, _ valid: [String]) throws -> Never {
    throw UsageError("\(what) \(value) does not exist (\(valid.joined(separator: ", ")))")
}

let usage = """
    usage: ingot-dev <command> [options]   (details of each command in docs/development.md)
    every command that loads a model needs --model DIR|org/repo (no default)
    accuracy (scripts/ci.sh --toy and --real):
      gen          --prompts 512,4K --n 128 [--check] [--mode both|spec|normal] [--jsonl f --label l] [--route-trace f]
      reuse        state after a reply and snapshots, bit-exact with the step-by-step path
      prefixcache  prefix cache across sessions, with ingot serve's defaults (lossy prefill on MoE; --int8 0 --qk8 0)
      splitcheck   [--prompt 4K] [--cuts 64,1024,…]   prefill split at 64-aligned cuts = whole, bit-exact
      shapes       [--models toy-a,toy-b,…] [--prompts 512]   several shapes in one process, each with its golden
      toymodels    --out DIR [--arch moe,dense,flash] [--seed 1] [--force]   toy models (scripts/ci.sh --toy)
      rowscheck    verification rows trimmed on the GPU (drafter) or k-row rounds = k steps (no drafter), bit-exact
      batchcheck   [--prompts 512,4K,chat] [--rows 2,4,8] [--n 32] [--reps R]   batched decode = alone, bit-exact
                   --serve [--rows 2,4] [--seed S]   ingot serve's loop with slots: each turn = alone, turn 2 from live
      fncheck      --model DIR --ref DIR [--layers N] [--gen N]   Flash-Next vs the harness (ref/flashnext/harness.py)
      fnprof       --model DIR [--rows 8] [--pos 4096] [--prefill 2048]       Flash-Next: per-piece kernel profile
    quality:
      kl           [--ref exact|mlx|mlx-f32|mlx-prefix]   KL against the exact path (default) or mlx-lm
    maintenance:
      rowscal      [--pdir prompts-es] [--n 256] [--cost-prompts 512,4K,16K,32K] [--out f]   row calibration
    --version      prints the version
    engine options: --int8 --qk8 --copy --tree N --spec-rows N --prep-trace --exp key=value
    in all: --pipelines FILE   appends to the file the kernels the command asked for
    """

runCommand {
    switch args.cmd {
    // Accuracy
    case "gen": try cmdGen()
    case "reuse": try cmdReuse()
    case "prefixcache": try cmdPrefixCache()
    case "splitcheck": try cmdSplitCheck()
    case "shapes": try cmdShapes()
    case "toymodels": try cmdToyModels()
    case "rowscheck": try cmdRowsCheck()
    case "batchcheck": try cmdBatchCheck()
    case "fncheck": try cmdFNCheck()
    case "fnprof": try cmdFNProf()
    // Quality
    case "kl":
        switch args.str("ref", "exact") {
        case "exact": try cmdKL()
        case "mlx", "mlx-f32": try cmdKLMLX()
        case "mlx-prefix": try cmdPFKL()
        case let r: try unknown("--ref", r, ["exact", "mlx", "mlx-f32", "mlx-prefix"])
        }
    // Maintenance
    case "rowscal": try cmdRowsCal()
    case "--version": print("ingot-dev \(IngotVersion.current)")
    default:
        print(usage)
        if args.cmd != "help" && args.cmd != "--help" { exit(2) }
    }
}
