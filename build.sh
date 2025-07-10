#!/usr/bin/env bash

# Build automation script for MSYS2/MinGW64 shell
# Supports: build, debug, clean, run

#build.sh commands:

#   Command	                What it does
#   ./build.sh build	    Release build
#   ./build.sh debug	    Debug build + launch gdb
#   ./build.sh run	        Run the built .exe
#   ./build.sh clean	    Remove build/ folder

BUILD_DIR=build
TARGET=test_tensor
GENERATOR="MinGW Makefiles"

usage() {
    echo "Usage: $0 [build|debug|clean|run]"
    exit 1
}

do_clean() {
    echo "[Clean] Removing build directory..."
    rm -rf "$BUILD_DIR"
    echo "[Clean] Done."
}

do_build() {
    echo "[Build] Configuring and building (Release)..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR" || exit 1

    cmake -G "$GENERATOR" -DCMAKE_BUILD_TYPE=Release ..
    cmake --build .

    cd ..
    echo "[Build] Done."
}

do_debug() {
    echo "[Debug] Configuring and building (Debug)..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR" || exit 1

    cmake -G "$GENERATOR" -DCMAKE_BUILD_TYPE=Debug ..
    cmake --build .

    cd ..
    echo "[Debug] Done. Launching gdb..."

    # Launch gdb if available
    if command -v gdb >/dev/null 2>&1; then
        gdb "$BUILD_DIR/$TARGET.exe"
    else
        echo "Error: gdb not found in PATH. Install it with pacman -S gdb."
        exit 1
    fi
}

do_run() {
    echo "[Run] Executing..."
    "./$BUILD_DIR/$TARGET.exe"
}

# Command dispatcher
if [ $# -eq 0 ]; then
    usage
fi

case "$1" in
    build)
        do_build
        ;;
    debug)
        do_debug
        ;;
    clean)
        do_clean
        ;;
    run)
        do_run
        ;;
    *)
        usage
        ;;
esac

