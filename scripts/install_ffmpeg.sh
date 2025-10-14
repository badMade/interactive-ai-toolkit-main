#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[install_ffmpeg] %s\n' "$1"
}

fail() {
  printf '[install_ffmpeg] ERROR: %s\n' "$1" >&2
  exit 1
}

if command -v ffmpeg >/dev/null 2>&1; then
  log 'FFmpeg is already installed.'
  exit 0
fi

platform="$(uname -s 2>/dev/null || echo unknown)"
case "$platform" in
  Darwin)
    if ! command -v brew >/dev/null 2>&1; then
      fail 'Homebrew is required but was not found. Install it from https://brew.sh/.'
    fi
    log 'Detected macOS. Installing FFmpeg via Homebrew...'
    brew install ffmpeg
    ;;
  Linux)
    if [[ -r /etc/os-release ]]; then
      . /etc/os-release
    else
      fail 'Unable to detect Linux distribution (missing /etc/os-release).'
    fi
    distro_id="${ID:-}"
    distro_like="${ID_LIKE:-}"
    distro_id="${distro_id,,}"
    distro_like="${distro_like,,}"
    if [[ "$distro_id" != debian && "$distro_id" != ubuntu && "$distro_id" != pop && "$distro_id" != linuxmint && "$distro_like" != *debian* && "$distro_like" != *ubuntu* ]]; then
      fail 'This installer currently supports only Debian or Ubuntu-based distributions.'
    fi
    if ! command -v apt-get >/dev/null 2>&1; then
      fail 'apt-get was not found. Please install FFmpeg using your package manager.'
    fi
    if [[ "$EUID" -eq 0 ]]; then
      log 'Detected Debian/Ubuntu. Installing FFmpeg via apt-get...'
      apt-get update
      apt-get install -y ffmpeg
    else
      if ! command -v sudo >/dev/null 2>&1; then
        fail 'sudo is required to install packages. Please install FFmpeg manually.'
      fi
      log 'Detected Debian/Ubuntu. Installing FFmpeg via sudo apt-get...'
      sudo apt-get update
      sudo apt-get install -y ffmpeg
    fi
    ;;
  *)
    fail "Unsupported platform: ${platform}."
    ;;
esac

if command -v ffmpeg >/dev/null 2>&1; then
  log 'FFmpeg installation completed successfully.'
else
  fail 'FFmpeg installation did not complete successfully.'
fi
