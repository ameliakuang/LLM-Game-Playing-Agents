# This file credits to https://github.com/AgentOpt/Trace-Bench/blob/main/KernelBench/install.sh
# Original author: Allen Nie

#!/usr/bin/env sh
set -eu

# Determine repository root (parent of this script)
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# Ensure uv is installed
if command -v uv >/dev/null 2>&1; then
  echo "uv is already installed: $(command -v uv)"
else
  echo "uv not found; installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Ensure uv is available on PATH for this shell session
if ! command -v uv >/dev/null 2>&1; then
  if [ -f "$HOME/.local/bin/env" ]; then
    . "$HOME/.local/bin/env"
  elif [ -x "$HOME/.local/bin/uv" ]; then
    PATH="$HOME/.local/bin:$PATH"
    export PATH
  fi
  # refresh shell command hash, if supported
  hash -r 2>/dev/null || true
fi

# Locate uv binary (absolute path fallback)
if command -v uv >/dev/null 2>&1; then
  UV_BIN=$(command -v uv)
elif [ -x "$HOME/.local/bin/uv" ]; then
  UV_BIN="$HOME/.local/bin/uv"
else
  echo "uv not found after installation; ensure $HOME/.local/bin is on PATH"
  exit 1
fi

# Prepare external directory and clone OCAtari there
command -v git >/dev/null 2>&1 || { echo "git is required"; exit 1; }

EXTERNAL_DIR=${EXTERNAL_DIR:-"$SCRIPT_DIR/external"}
OC_Atari_DIR=${OC_Atari_DIR:-"$EXTERNAL_DIR/OC_Atari"}
OC_Atari_URL=${OC_Atari_URL:-"https://github.com/ameliakuang/OC_Atari.git"}

mkdir -p "$EXTERNAL_DIR"

if [ -d "$OC_Atari_DIR/.git" ]; then
  echo "Updating OC_Atari in $OC_Atari_DIR"
  if ! (
    git -C "$OC_Atari_DIR" fetch --prune
    git -C "$OC_Atari_DIR" pull --ff-only
  ); then
    echo "OC_Atari update failed; recloning..."
    rm -rf "$OC_Atari_DIR"
    git clone "$OC_Atari_URL" "$OC_Atari_DIR"
  fi
else
  echo "Cloning OC_Atari into $OC_Atari_DIR"
  git clone "$OC_Atari_URL" "$OC_Atari_DIR"
fi

# Install Python dependencies for this task using uv
echo "Installing Python dependencies with uv in $SCRIPT_DIR"
(cd "$SCRIPT_DIR" && "$UV_BIN" sync)
echo "Activating the Python environment at $SCRIPT_DIR/.venv"
. "$SCRIPT_DIR/.venv/bin/activate"

# Use uv to install upstream OC_Atari as an editable package if it provides pyproject.toml or setup.py
if [ -f "$OC_Atari_DIR/pyproject.toml" ] || [ -f "$OC_Atari_DIR/setup.py" ]; then
  echo "Installing/updating upstream OC_Atari package in editable mode with uv"
  "$UV_BIN" pip install -e "$OC_Atari_DIR"
fi

# If OC_Atari has a requirements.txt, install those requirements explicitly.
if [ -f "$OC_Atari_DIR/requirements.txt" ]; then
  echo "Installing OC_Atari requirements.txt into the uv environment"
  "$UV_BIN" pip install -r "$OC_Atari_DIR/requirements.txt"
  echo "Adding OC_Atari requirements.txt to this uv project (locked)"
  "$UV_BIN" add --group ocatari -r "$OC_Atari_DIR/requirements.txt"
fi

echo "OC_Atari setup complete."

ACTIVATE_CMD="source \"$SCRIPT_DIR/.venv/bin/activate\""
echo "To activate this environment later, run: $ACTIVATE_CMD"