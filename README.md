# Gran Turismo 7 native PC renderer

This branch runs the PS5 build of Gran Turismo 7 on PC through a native Vulkan
renderer. The game's own engine code builds every frame; the renderer executes
the resulting command streams directly on the PC GPU, using shaders translated
ahead of time to SPIR-V and pipelines prepared before launch. It does not
emulate the PS5 GPU.

## Architecture

- **Command execution.** Submitted AGC command buffers run on a native executor.
  Draws and dispatches are recorded into Vulkan batches that are cut only where
  the game observes GPU results. Fences and labels are released when their batch
  completes.
- **Shaders.** RDNA programs are translated to SPIR-V with GPU-resolved
  resources: descriptors are read on the GPU from the game's own memory through
  a two-level page table, with no CPU-side descriptor folding.
- **Memory and resources.** Game memory is imported into Vulkan. Textures and
  render targets become native images, created from the game's texture and
  render-target initializers and then detiled once. GDS and LDS scratch are
  device-local.
- **Pipeline preparation.** `agc_graphics_library_prepare` builds compute
  pipelines, graphics stage libraries and the recorded pipeline journal ahead of
  time into the persistent shader cache. Shaders the cache does not cover yet are
  compiled when first used, on background threads. Work is never discarded while
  compilation is pending.

## GT7 status (2026-10-17)

- Startup screens and the title-screen UI render natively. The 3D scene behind
  the title screen and in-race 3D output are not rendered yet.
- Recent builds measured 4–37 FPS on the title screen. Full speed has not been
  reached yet. The known costs are the size of the translated shader code
  (wave64 on 32-wide hardware, and exec-mask expansion) and pipelines that are
  still compiled at first use.
- Visual output has not been compared against PS5 reference captures.

## Usage

- `tools/run_gt7_native.sh GAME_DIRECTORY` builds the runtime, prepares the
  pipelines and launches the game. Compute requests come from the game
  directory. Graphics requests are recorded in `native-stage-requests` there, or
  in `APS5_NATIVE_REQUEST_DIR` when that is set.
- `agc_graphics_library_prepare --native-compute | --native-graphics |
  --native-journal DIRECTORY` runs a preparation step without launching the
  game. `agc_shader_replay` replays individual shaders.
- During normal play the renderer reports only errors. Diagnostics can be turned on:
  - `APS5_NATIVE_LOG=1`: backend log;
  - `APS5_NATIVE_STATS=1`: per-frame statistics;
  - `APS5_NATIVE_GPU_TIMES=1`: per-program GPU time.
- `APS5_EMULATED_GPU=1` selects the emulated GPU path. `APS5_NATIVE_SKIP_DRAWS=1`
  selects CPU-only profiling.

## Source layout

- `core/libs/prx/libSceAgcDriver/Native`: executor, device, images and engine hooks.
- `core/shader/recompiler`: shader translation, GPU-resolved resources and
  persistent caches.

This branch includes implementation source and compatibility stubs only. It does
not include game executables, firmware, proprietary libraries, shaders extracted
from the game, assets, keys, runtime captures, or generated shader caches.

---

# About

Tool for automatic executables porting to Linux and Windows.

[**Web converter**](https://boykopovar.github.io/AnyPS5/) - files stay on your device.

![anyps5_collage1](https://github.com/user-attachments/assets/ed2f0b2d-33d3-46d7-ab88-767ca872fc99)

Includes a [relinker](core/relinker) that converts executable to the target system's native format and implementations of [system prx libraries](core/libs/prx) suitable for dynamic linking. No emulation or separate runtime process.

[**Discord**](https://discord.gg/BHFztBPUe), [Usage](docs/user/USAGE.md), [Build instructions](docs/dev/BUILD.md), [Architecture](docs/dev/ARCHITECTURE.md), [Technical debt of the project](docs/dev/TechnicalDebt.md), [code style conventions](docs/dev/CONVENTIONS.md), [contributing](CONTRIBUTING.md)

## Status

[![libraries](https://boykopovar.github.io/AnyPS5/badge-libraries.svg)](https://boykopovar.github.io/AnyPS5/progress.html) [![shaders](https://boykopovar.github.io/AnyPS5/badge-shaders.svg)](https://boykopovar.github.io/AnyPS5/progress.html)

[![progress map](https://boykopovar.github.io/AnyPS5/progress.svg)](https://boykopovar.github.io/AnyPS5/progress.html)

[Progress history](https://boykopovar.github.io/AnyPS5/history.html)

<sub>* System libraries: percentage of the functions known to the project so far (declared in [core/libs/prx](core/libs/prx)), not of every PS5 system function. The total grows as more functions are declared.</sub>

Unsupported or unexpected states strictly throw `std::runtime_error`. `what()` is printed to stderr and the process terminates.

The [shader recompiler](core/shader/recompiler/Recompiler.cpp) successfully produces SPIR-V (validated via [Spirv-Tools](3rdparty/SPIRV-Tools) when built with `ANYPS5_ENABLE_SPIRV_TOOLS`).

## Compatibility

Per-game progress (stage reached, open blockers, libraries and GPU coverage) is tracked on the [blocker map](https://alxcbtw.github.io/anyps5-roadmap/), a community project built from public GitHub data ([source](https://github.com/AlxcBTW/anyps5-roadmap)). Test reports go in the [Game bring-up](https://github.com/boykopovar/AnyPS5/discussions/categories/game-bring-up) discussions, one thread per title.

## Input mapping

SDL-mapped game controllers are supported, including analog sticks and triggers. Keyboard and mouse controls can be configured with an `anyps5-input.ini` file. See [input mapping](docs/user/INPUT_MAPPING.md) for the supported devices and configuration format.

## Disclaimer

This project is intended for interoperability, research, preservation, and compatibility purposes. It does not include, distribute, or require copyrighted software, firmware, cryptographic keys, or proprietary libraries. Users are responsible for ensuring that any binaries used with this project are obtained and used in accordance with applicable laws and their respective license terms.

## License

This project is licensed under the GNU General Public License version 2 only.
