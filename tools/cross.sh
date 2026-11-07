#!/usr/bin/env bash
# Cross-compile for a board, copy the binaries over ssh and run them there.
#
#   tools/cross.sh BOARD test [CARGO_TEST_ARGS] [-- TEST_ARGS]
#   tools/cross.sh BOARD run  --example NAME [-- ARGS]
#   tools/cross.sh BOARD copy --example NAME
#
# BOARD:
#   pi4   Raspberry Pi 4 (Cortex-A72), ssh host Pi4
#   orin  Jetson Orin Nano (Cortex-A78AE), ssh host LabOrinWiFi
#
# Examples:
#   tools/cross.sh pi4 test                              # every test, on the Pi
#   tools/cross.sh pi4 test --lib klt                    # the lib tests matching "klt"
#   tools/cross.sh orin test --lib -- --ignored tests::test_
#                     # the GPU tests on the Orin: their test_ wrappers, which
#                     # run each inner_ test in a fresh process
#   tools/cross.sh pi4 run --example bench_euroc_cpu -- ~/Downloads/V1_01_easy/
#   tools/cross.sh pi4 copy --example euroc_live         # then run it on the Pi's desktop
#
# Release builds with frame pointers (for samply), one target directory per
# board (target/cross/BOARD) since their RUSTFLAGS differ, linked against
# glibc 2.35 (the Orin's JetPack 6; the Pi's Arch Linux ARM is newer).
#
# Binaries land in RV_CROSS_DIR on the board (default /tmp/rudolf-v-cross),
# named without cargo's hash so reruns overwrite them. Debug info is stripped
# before copying (symbols stay, for backtraces); RV_CROSS_STRIP=0 keeps it,
# for samply's source lines. RV_CROSS_HOST overrides the ssh host.
#
# Needs: rustup target add aarch64-unknown-linux-gnu; cargo install
# cargo-zigbuild; zig and llvm-strip on PATH.
set -euo pipefail

# One shared ssh connection for every copy and run: the Pi's ufw rate-limits
# new ssh connections (6 in 30 s), and a test run makes two per binary.
ssh_opts=(-o ControlMaster=auto -o "ControlPath=${XDG_RUNTIME_DIR:-/tmp}/rv-cross-%C"
    -o ControlPersist=60)

# Runner mode: cargo calls this script with a built binary and its arguments.
if [[ -n "${RV_CROSS_RUNNER:-}" ]]; then
    bin="$1"
    shift
    name="$(basename "$bin")"
    [[ "$name" =~ ^(.*)-[0-9a-f]{16}$ ]] && name="${BASH_REMATCH[1]}"
    src="$bin"
    if [[ "${RV_CROSS_STRIP:-1}" != 0 ]]; then
        src="$bin.stripped"
        llvm-strip --strip-debug -o "$src" "$bin"
    fi
    scp -q "${ssh_opts[@]}" "$src" "$RV_CROSS_HOST:$RV_CROSS_DIR/$name"
    if [[ "$RV_CROSS_RUNNER" == copy ]]; then
        echo "copied to $RV_CROSS_HOST:$RV_CROSS_DIR/$name"
        exit 0
    fi
    args=""
    (($#)) && args="$(printf ' %q' "$@")"
    exec ssh "${ssh_opts[@]}" "$RV_CROSS_HOST" "cd $RV_CROSS_DIR && chmod +x $name && ./$name$args"
fi

usage="usage: tools/cross.sh pi4|orin test|run|copy [cargo args] [-- args]"
board="${1:?$usage}"
mode="${2:?$usage}"
shift 2
case "$board" in
    pi4) host=Pi4 cpu=cortex-a72 ;;
    orin) host=LabOrinWiFi cpu=cortex-a78ae ;;
    *) echo "$usage" >&2; exit 2 ;;
esac
case "$mode" in
    test) cmd=test runner=run ;;
    run) cmd=run runner=run ;;
    copy) cmd=run runner=copy ;;
    *) echo "$usage" >&2; exit 2 ;;
esac

self="$(realpath "$0")"
cd "$(dirname "$self")/.."
export RV_CROSS_RUNNER="$runner"
export RV_CROSS_HOST="${RV_CROSS_HOST:-$host}"
export RV_CROSS_DIR="${RV_CROSS_DIR:-/tmp/rudolf-v-cross}"
ssh "${ssh_opts[@]}" "$RV_CROSS_HOST" "mkdir -p $RV_CROSS_DIR"

CARGO_TARGET_DIR="target/cross/$board" \
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="$self" \
RUSTFLAGS="-C target-cpu=$cpu -C force-frame-pointers=yes" \
    exec cargo-zigbuild "$cmd" --release --target aarch64-unknown-linux-gnu.2.35 "$@"
