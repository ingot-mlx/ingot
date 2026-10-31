# Changelog

All notable changes to ingot, newest first. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and ingot follows [Semantic Versioning](https://semver.org/)
in the 0.x series (see "Versions and releases" in `docs/development.md`).

## [Unreleased]

### Added

- Install with Homebrew: `brew install ingot-mlx/tap/ingot` (tap `ingot-mlx/homebrew-tap`, with a bottle for Apple
  silicon on macOS 27); `brew upgrade ingot` updates it. Building from source still works.

## [0.1.0] - 2026-10-31

First tagged release; changes since the initial public release of 2026-10-30.

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

- `ingot --version` and `ingot-dev --version`; `GET /health` returns the version (`{"status":"ok","version":"0.1.0"}`)
  and `ingot serve` prints it on its first line.
- DFlash v1 drafters with head 64, q/k/v/o biases and YaRN, used by gpt-oss 20b.

[Unreleased]: https://github.com/ingot-mlx/ingot/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ingot-mlx/ingot/releases/tag/v0.1.0
