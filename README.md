# GT7 native Vulkan work in progress

This branch contains experimental Gran Turismo 7 (PS5) bring-up and a native Vulkan
backend for a relinked executable. Native rendering is the default on this GT7 branch.
It is unfinished, is not an official PC port,
and does not yet provide full-speed gameplay or verified PS5-equivalent graphics.

## Current progress (2026-10-17)

- Native Vulkan renders the startup screens and parts of the title-screen UI.
  The 3D background remains black, and some graphics are incomplete.
- Recent title-screen runs completed 600 seconds without a failed GPU batch,
  at approximately 4–5 FPS. These runs do not establish race compatibility.
- Draws and dispatches wait for first-use shader and pipeline compilation rather
  than being discarded while compilation is pending. Persistent shader caches
  and offline replay/preparation tools support cache warm-up.
- The backend consumes submitted AGC command state and executes translated RDNA
  shaders as SPIR-V on Vulkan. It includes GPU descriptor lookup, imported guest
  buffers, native images, device-local GDS, indirect dispatch, and resource hooks.
  Memory aliasing/coherency and synchronization remain active development areas.
- All 232 available captured compute requests now translate offline. Selected
  previously failing modules pass SPIR-V validation. Captures are not included.
  The latest device-local LDS scratch and sampler-capacity changes build but
  have not yet been checked in-game.
- Known issues include pixel shaders reaching the loop guard, missing 3D output,
  shader expansion/performance, and incomplete graphics validation. No PS5
  reference captures have been used to establish visual identity.

## Development entry points

- `core/libs/prx/libSceAgcDriver/Native`: native execution, device, images, and hooks.
- `core/shader/recompiler`: GPU-resolved resources, runtime shader addresses,
  shader translation, and persistent caches.
- `agc_shader_replay` and `agc_graphics_library_prepare`: offline shader tools.
- The experimental GT7 native backend is the default on this branch.
  `APS5_EMULATED_GPU=1` selects the alternate emulated GPU path;
  `APS5_NATIVE_SKIP_DRAWS=1` selects CPU-only profiling.
  `APS5_NATIVE_STATS=1` and `APS5_NATIVE_GPU_TIMES=1` enable diagnostics.

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
