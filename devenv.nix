{ pkgs, lib, config, inputs, ... }:

{
  # https://devenv.sh/packages/
  packages = [
    pkgs.git
    pkgs.pico-sdk
    pkgs.picotool
    pkgs.cmake
    # pkgs.gcc
    pkgs.clang-tools
    pkgs.gcc-arm-embedded
    pkgs.picotool
    pkgs.minicom
    pkgs.ninja
];

  languages = {
    python = {
      enable = true;
      package = pkgs.python313;
    };
    c = {
      enable = true;
    };
  };

  # https://devenv.sh/languages/
  # languages.rust.enable = true;

  # https://devenv.sh/processes/
  # processes.dev.exec = "${lib.getExe pkgs.watchexec} -n -- ls -la";

  # https://devenv.sh/services/
  # services.postgres.enable = true;


  scripts.open-serial-console.exec = ''
    minicom --device /dev/ttyACM0
  '';

  scripts.update-odt.exec = ''
    set -euo pipefail
    ODT_DIR="OnDeviceTraining/src"
    if [ ! -d "$ODT_DIR/.git" ]; then
      echo "error: $ODT_DIR is not a git checkout. Run 'cmake --preset PREPARE' first, or invoke from repo root." >&2
      exit 1
    fi
    if [ -n "$(git -C "$ODT_DIR" status --porcelain)" ]; then
      echo "error: local changes in $ODT_DIR — stash or commit first:" >&2
      git -C "$ODT_DIR" status --short >&2
      exit 1
    fi
    old_sha=$(git -C "$ODT_DIR" rev-parse --short HEAD)
    git -C "$ODT_DIR" fetch origin develop
    git -C "$ODT_DIR" reset --hard origin/develop
    new_sha=$(git -C "$ODT_DIR" rev-parse --short HEAD)
    if [ "$old_sha" = "$new_sha" ]; then
      echo "vendored ODT already at upstream develop ($new_sha)."
    else
      echo "vendored ODT: $old_sha -> $new_sha"
      git -C "$ODT_DIR" log --oneline "$old_sha..$new_sha"
    fi
  '';

  # See full reference at https://devenv.sh/reference/options/
}
