#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
    echo "Usage: $0 GAME_DIRECTORY [game arguments...]" >&2
    exit 2
fi
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
game_dir="$(cd -- "$1" && pwd)"
shift
build_dir="$repo_dir/build"
runtime_libs="$build_dir/core/libs/libs"
preparer="$build_dir/core/libs/prx/libSceAgcDriver/agc_graphics_library_prepare"
[[ -x "$game_dir/gt7ps5.elf" ]] || { echo "Missing gt7ps5.elf in $game_dir" >&2; exit 1; }

cmake --build "$build_dir" --target libs agc_graphics_library_prepare --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-16}"
cache_dir="${ANYPS5_SHADER_CACHE_DIR:-$game_dir/shader_cache}"
export ANYPS5_SHADER_CACHE_DIR="$cache_dir"
if [[ -z "${APS5_EMULATED_GPU:-}" || "${APS5_EMULATED_GPU:-0}" == 0 ]] &&
   [[ -z "${ANYPS5_NO_SHADER_CACHE:-}" || "${ANYPS5_NO_SHADER_CACHE:-0}" == 0 ]] && compgen -G "$game_dir/shader_*.req" > /dev/null; then
    echo "Preparing captured native compute pipelines before launch..."
    env -u LD_PRELOAD -u LD_LIBRARY_PATH "$preparer" --native-compute "$game_dir" > "$game_dir/native-compute-prewarm.log" 2>&1
fi

native_requests="${APS5_NATIVE_REQUEST_DIR:-$game_dir/native-stage-requests}"
export APS5_NATIVE_REQUEST_DIR="$native_requests"
if [[ -z "${APS5_EMULATED_GPU:-}" || "${APS5_EMULATED_GPU:-0}" == 0 ]] &&
   [[ -z "${ANYPS5_NO_SHADER_CACHE:-}" || "${ANYPS5_NO_SHADER_CACHE:-0}" == 0 ]] && compgen -G "$native_requests/*.req" > /dev/null; then
    echo "Preparing captured native graphics libraries before launch..."
    env -u LD_PRELOAD -u LD_LIBRARY_PATH "$preparer" --native-graphics "$native_requests" > "$game_dir/native-graphics-prewarm.log" 2>&1
fi

if [[ -z "${APS5_EMULATED_GPU:-}" || "${APS5_EMULATED_GPU:-0}" == 0 ]] &&
   [[ -z "${ANYPS5_NO_SHADER_CACHE:-}" || "${ANYPS5_NO_SHADER_CACHE:-0}" == 0 ]] &&
   [[ -z "${APS5_NO_PIPELINE_JOURNAL:-}" || "${APS5_NO_PIPELINE_JOURNAL:-0}" == 0 ]]; then
    echo "Preparing recorded native pipeline recipes before launch..."
    env -u LD_PRELOAD -u LD_LIBRARY_PATH APS5_PIPELINE_JOURNAL_THREADS="${APS5_PIPELINE_JOURNAL_THREADS:-4}" \
        "$preparer" --native-journal > "$game_dir/native-journal-prewarm.log" 2>&1
fi

export LD_LIBRARY_PATH="$runtime_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$runtime_libs/libGt7MissingImports.prx${LD_PRELOAD:+:$LD_PRELOAD}"
cd -- "$game_dir"
exec ./gt7ps5.elf "$@" > "${GT7_PS5_LOG:-$game_dir/gt7ps5.log}" 2>&1
