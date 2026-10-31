# Changelog

Changes since the initial public release, newest first. There are no tagged releases yet.

## Unreleased

### Changed

- Everything is now called ingot: the binaries are `ingot` and `ingot-dev` (were `lab` and `lab-dev`), the targets
  `Ingot`, `IngotServe`, `IngotCLI` and `IngotBench`, the environment variables `INGOT_MODELS` and
  `INGOT_SKIP_CHECK`, and the cache `~/.cache/ingot`. There is no backward compatibility. Flash-Next models converted
  before the rename need the `metal_lab` block of their `config.json` renamed to `ingot`.
- gpt-oss 20b loads z-lab's DFlash v1 drafter (`z-lab/gpt-oss-20b-DFlash`) by default; `--drafter none` turns it
  off, and the 120b runs without a drafter. Tokens are the same with and without the drafter.
- `ingot serve` and `ingot run` fail with the `hf download` line to fetch a family's default drafter when it is not
  in the Hugging Face cache, instead of starting without it.

### Added

- DFlash v1 drafters with head 64, q/k/v/o biases and YaRN, used by gpt-oss 20b.
