# Third-party notices

This project is MIT-licensed (see `LICENSE`). It contains code and data that come from, or were adapted from, the
projects below. Each entry gives what was taken, where it lives in this tree, the origin, the licence (checked at the
source on 2026-10-28) and the notice the licence requires. The full texts of the MIT licences that require a notice are
at the end of this file; the Apache-2.0 text is in `LICENSES/Apache-2.0.txt`.

Classes: **copied** (literal or near-literal fragments), **adapted** (logic or structure ported to Swift/Metal, our own
code), **idea** (an algorithm or semantics reimplemented from the description) and **format** (we read a format they
define). Entries of class idea or format carry no obligation and are credited as a courtesy. No model weights are
included.

## Copied and adapted code

### Strata (Niko1221/Strata) — MIT, Copyright (c) 2026 Niko1221 and the Strata contributors

- `Sources/MetalLab/CopySpec.swift`: `CopyIndex` and `CopyPolicy`. **Copied** (a near-literal translation from C++:
  hash mix and constants, 4-way index, load bound, hit-rate priors, decay and thresholds of the policy).
- Origin: `src/spec/suffix_drafter.cpp`, `suffix_drafter.hpp` and `src/spec/draft_policy.cpp` at commit `e88ee39`.
- Notice: MIT text below ("Strata"). Strata's licence is MIT and has always been; the earlier comment in the source
  said Apache-2.0 by mistake.

### MLX (ml-explore/mlx) — MIT, Copyright (c) 2023 Apple Inc.

- **Copied:** `nax_coord` and `nax_mma2` in `Sources/MetalLab/Shaders/qmm_nax.metal` (from `steel/gemm/nax.h`, fragment
  coordinates and the 16×32×16 matmul; largely dictated by the hardware fragment layout) and `lab_fp4x2` in
  `Sources/MetalLab/Shaders/qmm_nax_moe.metal` (from `fp_quantized.h` and `fp_quantized_nax.h`, one expression with the
  masks and the 2^14 factor). Origin commit `7d2d94a`.
- **Adapted:** `qmm_t_nax` in `qmm_nax.metal` (scheme of `quantized_nax.h`, `qmm_t_nax_tgp_impl`) and the mask trick of
  `vload_x`, `vqdot` and `qmv_acc` in `Sources/MetalLab/Shaders/verify.metal` (from `quantized.h`, `load_vector` and
  `qdot`).
- **Idea / format:** affine and mxfp4 quantization formats (`Sources/MetalLab/Loader/Quant.swift`, `qmm.metal`) and
  `steel_attention_nax` (`pattn64_i8` in `verify_attn_pf.metal`, which reuses `nax_coord` and `nax_mma2`).
- The "0x6400" half-precision bit trick used in several kernels is **not** from MLX (it is not in MLX's tree); it is a
  well-known technique and carries no notice.
- Notice: MIT text below ("Apple").

### mlx-lm (ml-explore/mlx-lm) — MIT, Copyright (c) 2023 Apple Inc.

- **Adapted:** YaRN frequencies in `Sources/MetalLab/Loader/RopeTable.swift` (`rope_utils.py`, commit `1f7057c`);
  tool-call format detection in `Sources/LabServe/ChatDialect.swift` (`tokenizer_utils.py`, `_infer_tool_parser`, v0.32.2);
  tool-call parsers in `Sources/LabServe/ToolCallBlocks.swift` and `HeaderChannels.swift` (`tool_parsers/glm47.py`,
  `json_tools.py`, `gemma4.py`, `muse_spark.py`, v0.32.2).
- **Idea:** model semantics (router, `lfm2_moe`, group selection, proportional RoPE) in `qmm_nax_moe.metal`,
  `ab_moe_route.metal`, `yw_lfm.metal`, `ShortConv.swift` and `RopeTable.swift`. The Python tools under `ref/` import
  `mlx_lm` and `mlx_vlm` as dependencies; nothing of them is vendored.
- Notice: MIT text below ("Apple", the same text as MLX).

### splash (incoai/splash) — Apache-2.0 (no copyright line in its LICENSE; no NOTICE file)

- **Adapted:** `Sources/LabServe/OutputParser.swift`, a modified Swift port of `server/output.py` at tag 1.3.2
  (holding back partial tags, value conversion by schema, reasoning split, tool-call state machine). Design adapted
  (own code): `moe_group` in `qmm_nax_moe.metal` (`kernels/shared/moe.metal`) and the GDN "tape" in `verify.metal` and
  `Layers/GDN.swift` (`decode/gdn.metal`).
- **Idea:** the other mentions of splash in the comments (`verify_nax.metal`, `verify_attn_pf.metal`,
  `verify_attn_nax.metal`, `yy_model.metal`, `Drafter.swift`, `zg_qsa.metal`, `zf_flashnext.metal`, `ExpertStream.swift`,
  `ExpertWide.swift`, `FlashPLERows.swift`, `Layers/MoE.swift`, `Memory/KVPool.swift`, `scripts/flashnext/fnformat.py`).
- Notice (Apache-2.0 §4a–c): licence copy in `LICENSES/Apache-2.0.txt`; the changed files say so in a header line
  (`Portions adapted from splash (Apache-2.0); changed: …`); the attribution is kept. §4d does not apply (no NOTICE).

### oMLX (jundot/omlx) — Apache-2.0, Copyright 2025 oMLX contributors (no NOTICE file)

- **Adapted (design):** single-expert tiles with descriptors written on the GPU, `row_map`, SwiGLU in the epilogue, in
  `Sources/MetalLab/Shaders/qmm_nax_moe.metal` (`patches/m5_gather_qmm_nax.py`, `moe_routes.py`, `quantized_moe.h`
  at v0.7.1rc1).
- **Idea:** `verify.metal`, `Layers/GDN.swift`, `Engine.swift`.
- Notice: as for splash.

### TensorFold (ashhart/TensorFold) — Apache-2.0 from 0.6.0 (MIT up to 0.5.0)

- **Adapted (design, at the border of idea):** the slot stack of `gdn_recur_tree` in
  `Sources/MetalLab/Shaders/zt_tree.metal`, from `zig/kernels/metal/core/deltanet_slots.metal` (entered 2026-10-17,
  already under Apache-2.0). **Idea:** `zg_qsa.metal` (QSA).
- Notice (Apache-2.0 §4a–d, applied out of prudence): licence copy in `LICENSES/Apache-2.0.txt`, a header line in
  `zt_tree.metal`, and the text of its NOTICE, reproduced here:

  > TensorFold
  > Copyright 2026 TensorFold contributors
  > https://github.com/ashhart/TensorFold
  >
  > TensorFold is licensed under the Apache License, Version 2.0 (LICENSE) from 0.6.0.
  > Releases up to 0.5.0 were MIT-licensed, and code written before 0.6.0 keeps its
  > MIT notice in LICENSES/MIT.txt. Third-party code and its licenses are listed in
  > THIRD_PARTY_NOTICES.md.

### mlx-serve (ddalcu/mlx-serve) — MIT, Copyright (c) 2026 David Dalcu

- **Adapted (protocol only, no code; Zig versus Swift/Metal):** the DFlash2 drafter protocol and shape in
  `Sources/MetalLab/Drafter.swift` and `Shaders/yy_model.metal` (`src/dflash.zig`). **Idea:** fused qkv in
  `Layers/Linear.swift`. MIT text below, out of prudence.

### Chat template of the toy models

- `ref/toy/chat_template.jinja` is a **literal copy** (11,269 bytes, sha256 `94e94428…1ad4`) of the template in
  `froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-4bit` (revision `32939a6c`, licence apache-2.0 on its model card,
  no LICENSE file or credits), which rewrites the template of `Qwen/Qwen3.6-35B-A3B` (Apache-2.0, Copyright 2026
  Alibaba Cloud). Unchanged, so no change notice. Licence copy: `LICENSES/Apache-2.0.txt`.

## Ideas, formats and dependencies (courtesy credits)

| Project | Licence | Used for |
|---|---|---|
| mlx-vlm (Blaizzy/mlx-vlm) | MIT, Copyright (c) 2025 Prince Canuma | semantics of `qwen4_exp` (`zf_flashnext.metal`, `zg_qsa.metal`); imported by `ref/flashnext/harness.py` |
| flash-linear-attention (fla-org) | MIT, Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li | WY/UT form of the chunked gated delta rule (`yz_gdn_chunk.metal`) |
| DFlash (z-lab/dflash) | MIT, Copyright (c) 2026 Z Lab | DFlash v1 semantics (`Drafter.swift`, `yy_model.metal`) |
| llama.cpp / ggml | MIT, Copyright (c) 2023-2026 The ggml authors | ideas (`StreamKeepAlive.swift`, `ChatTemplate.swift`, `CheckSupport.swift`); GGUF `block_q2_0` read by our own reader in `scripts/flashnext/fnformat.py` |
| Prism ML / Bonsai 2 (`prism_hadamard_qwen35`) | models Apache-2.0; NOTICE: Copyright 2026-present Prism ML, Inc. | ternary weight format read by `xc_tern.metal`, `Layers/Linear.swift`, `Loader/ModelSpec.swift`; no code taken |
| vLLM | Apache-2.0 (Copyright contributors to the vLLM project) | sampling and penalty semantics, `qwen3_coder` XML format (`Sampling.swift`, `zz_sample.metal`, `OutputParser.swift`, `ref/penalty_ref.py`) |
| SGLang | Apache-2.0 (Copyright 2023-2024 SGLang Team) | deterministic QSA semantics (`zg_qsa.metal`), negative-control idea (`scripts/ci/toys.sh`) |
| Rapid-MLX | Apache-2.0 (Copyright 2025-2026 the Rapid-MLX contributors) | shared-prefix boundary idea (`PrefixCache.swift`) |
| swift-jinja (Hugging Face), swift-collections (Apple) | Apache-2.0 | SwiftPM dependencies, not vendored (they oblige only if binaries are distributed) |

## Model configurations and test fixtures

`ref/toy/real/*.json` are trimmed `config.json` files (the key `metal_lab_real` records the source and date) and
`Tests/LabServeTests/Resources/*/` hold chat-template outputs, token ids and special tokens of each tokenizer. They are
small factual data, not the models, and are included with attribution to the models they describe. Licence as stated on
the original model card (checked on the Hugging Face API on 2026-10-29 or, for the first group, 2026-10-28):

| Model (source of the config / fixtures) | Licence |
|---|---|
| Qwen3.5-0.8B, Qwen3.5-2B, Qwen3.6-35B-A3B (Heretic), Qwen3.8-27B (Qwen, Alibaba Cloud); froggeric's Heretic fixtures | Apache-2.0 |
| gpt-oss-20b (OpenAI), Granite 4.1 3B (IBM), Muse Glimmer 30B (Meta models) | Apache-2.0 |
| Gemma 4 12B, 26B-A4B and 31B (Google) | Apache-2.0 |
| Gemma 4 E2B and E4B (Google) | Gemma terms of use |
| GLM-4.7-Flash (Z.ai) | MIT |
| LFM2.5-8B-A1B, LFM2.5-24B-A2B, LFM3-1.7B (Liquid AI) | LFM Open License v1.0 (`lfm1.0`) |
| Nemotron 3 Nano 4B (NVIDIA) | NVIDIA Nemotron Open Model License |
| Nemotron 3.5 Lightning 30B-A3B (NVIDIA) | OpenMDW 1.1 (`other` on the card) |
| Qwen3.8-Flash-Next | `other` on the card |

The quantized copies come from `mlx-community` and `LiquidAI` (`source` in each config). The licences of the last five
rows cover "the model and its materials"; only trimmed configuration and output data are included. Remove them if the
terms of a row are not acceptable.

## License texts

### Apple (MLX, mlx-lm)

    MIT License

    Copyright © 2023 Apple Inc.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

### Strata

    MIT License

    Copyright (c) 2026 Niko1221 and the Strata contributors

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

### mlx-serve

    MIT License

    Copyright (c) 2026 David Dalcu

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
