#!/usr/bin/env bash

# ---------------------------------
# Automated build script
# Supports: build, debug, clean, run
# ---------------------------------

set -e

BUILD_DIR=build
TARGET=test_tensor
GENERATOR="MinGW Makefiles"

COLOR_RESET="\033[0m"
COLOR_GREEN="\033[0;32m"
COLOR_YELLOW="\033[1;33m"
COLOR_RED="\033[0;31m"
COLOR_BLUE="\033[1;34m"

usage() {
    echo -e "${COLOR_YELLOW}Usage: $0 [build|debug|clean|run]${COLOR_RESET}"
    echo
    echo "  build    Configure and compile in Release mode"
    echo "  debug    Configure and compile in Debug mode, launch gdb"
    echo "  clean    Remove build directory"
    echo "  run      Execute the built binary"
    exit 1
}

do_clean() {
    echo -e "${COLOR_BLUE}[Clean] Removing build directory...${COLOR_RESET}"
    rm -rf "$BUILD_DIR"
    echo -e "${COLOR_GREEN}[Clean] Done.${COLOR_RESET}"
}

do_configure_and_build() {
    local build_type=$1
    echo -e "${COLOR_BLUE}[Build] Configuring (${build_type})...${COLOR_RESET}"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    cmake -G "$GENERATOR" -DCMAKE_BUILD_TYPE=$build_type ..
    cmake --build .

    cd ..
    echo -e "${COLOR_GREEN}[Build] Done.${COLOR_RESET}"
}

do_build() {
    do_configure_and_build "Release"
}

do_debug() {
    do_configure_and_build "Debug"
    echo -e "${COLOR_BLUE}[Debug] Launching gdb...${COLOR_RESET}"
    if command -v gdb >/dev/null 2>&1; then
        gdb "$BUILD_DIR/$TARGET.exe"
    else
        echo -e "${COLOR_RED}[Error] gdb not found in PATH.${COLOR_RESET}"
        exit 1
    fi
}

do_run() {
    echo -e "${COLOR_BLUE}[Run] Executing...${COLOR_RESET}"
    "./$BUILD_DIR/$TARGET.exe"
}

# Main command dispatcher
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
