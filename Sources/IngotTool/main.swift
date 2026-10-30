import Foundation
import Ingot
import IngotCLI

// ingot: use the engine (OpenAI-compatible server, model info, memory plan and a single prompt).
// Correctness checks and benchmarks live in ingot-dev. Usage: see the README.

runCommand {
    switch args.cmd {
    case "serve": try cmdServe()
    case "info": try cmdInfo()
    case "plan": try cmdPlan()
    case "run": try cmdRun()
    default:
        print(
            """
            usage: ingot <command> [options]   (details in the README)
              serve   --model org/repo|dir [--port 8090] [--name id] [--max-ctx N] ...   OpenAI-compatible API
              run     --model org/repo|dir --prompt "text" [--max-tokens N] [--no-think]  a single prompt
              plan    [--model org/repo|dir] [--mem-limit-gb N]   memory plan without loading the model
                      (without --model: every model in $INGOT_MODELS)
              info    --model org/repo|dir   model files, config and quantization
            engine options: --int8 0|1 --qk8 0|1 --copy auto|on|off --exp key=value
            on all commands: --pipelines FILE   appends the requested kernels to the file
            correctness checks and benchmarks: ingot-dev
            """)
        if args.cmd != "help" && args.cmd != "--help" { exit(2) }
    }
}
