#!/usr/bin/env bash
set -euo pipefail

print_msg() {
  local level="$1"
  shift
  printf '%s\n' "$level: $*" >&2
}

ensure_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    print_msg "ERROR" "Required command '$cmd' not found. Please install it and re-run this installer."
    exit 1
  fi
}

run_with_privilege() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    ensure_command sudo
    sudo "$@"
  fi
}

install_with_apt() {
  ensure_command apt-get
  print_msg "INFO" "Updating package index via apt-get..."
  run_with_privilege apt-get update
  print_msg "INFO" "Installing FFmpeg via apt-get..."
  run_with_privilege apt-get install -y ffmpeg
}

main() {
  local platform
  platform="$(uname -s)"

  case "$platform" in
    Darwin)
      ensure_command brew
      print_msg "INFO" "Installing FFmpeg via Homebrew..."
      brew install ffmpeg
      ;;
    Linux)
      if [ -f /etc/os-release ]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        case "${ID:-}" in
          debian|ubuntu)
            install_with_apt
            ;;
          *)
            print_msg "ERROR" "Unsupported Linux distribution '$ID'. Please install FFmpeg manually."
            exit 1
            ;;
        esac
      else
        print_msg "ERROR" "Unsupported Linux distribution. /etc/os-release not found."
        exit 1
      fi
      ;;
    *)
      print_msg "ERROR" "Unsupported platform '$platform'. Please install FFmpeg manually."
      exit 1
      ;;
  esac

  print_msg "INFO" "FFmpeg installation completed successfully."
}

main "$@"
