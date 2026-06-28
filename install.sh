#!/usr/bin/env bash
#
# install.sh - PyClaw installation script for Linux (Debian/Ubuntu, WSL,
# and other Linux distributions with python3 and git available).
#
# What this does:
#   1. Clones the PyClaw repository (skipped if already present)
#   2. Creates an isolated Python virtual environment (.venv)
#   3. Installs dependencies from requirements.txt into that environment
#   4. Prints the exact commands to start PyClaw
#
# Usage:
#   ./install.sh [install-directory]
#
# If install-directory is omitted, PyClaw is installed into ./pyclaw
# relative to wherever you run this script from.
#
# This script does NOT install or configure an LLM backend (llama.cpp,
# Ollama, etc.) -- see README.md for that. It only sets up PyClaw itself.

set -euo pipefail

REPO_URL="${PYCLAW_REPO_URL:-https://github.com/your-org/pyclaw.git}"
INSTALL_DIR="${1:-pyclaw}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
info()  { printf '\033[1;36m==>\033[0m %s\n' "$1"; }
warn()  { printf '\033[1;33m==>\033[0m %s\n' "$1"; }
error() { printf '\033[1;31m==>\033[0m %s\n' "$1" >&2; }

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        error "Required command '$1' was not found on PATH."
        error "Install it first (e.g. 'sudo apt install $1' on Debian/Ubuntu), then re-run this script."
        exit 1
    fi
}

# ----------------------------------------------------------------------
# Preflight checks
# ----------------------------------------------------------------------
require_command "$PYTHON_BIN"
require_command git

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
info "Using $PYTHON_BIN (version $PYTHON_VERSION)"

PYTHON_MAJOR="$("$PYTHON_BIN" -c 'import sys; print(sys.version_info[0])')"
PYTHON_MINOR="$("$PYTHON_BIN" -c 'import sys; print(sys.version_info[1])')"
if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]; }; then
    warn "PyClaw targets Python 3.11+. Detected $PYTHON_VERSION -- this may still work, but isn't tested."
fi

# ----------------------------------------------------------------------
# Detect: are we already standing inside a PyClaw checkout?
# ----------------------------------------------------------------------
# This covers being run from inside a real `git clone` (.git present) AND
# from inside a checkout obtained another way -- e.g. extracted from a
# downloaded zip -- which has no .git folder but does have PyClaw's own
# files. Checking only for .git would otherwise try to re-clone over a
# directory the user is already sitting in, which fails outright if
# REPO_URL hasn't been pointed at a real repository yet.
looks_like_pyclaw_checkout() {
    [ -f "$1/main.py" ] && [ -f "$1/config.py" ] && [ -d "$1/agent" ]
}

# ----------------------------------------------------------------------
# Clone (or reuse an existing checkout)
# ----------------------------------------------------------------------
if looks_like_pyclaw_checkout "."; then
    info "Already inside a PyClaw checkout -- skipping clone."
    INSTALL_DIR="."
elif [ -d "$INSTALL_DIR/.git" ] || looks_like_pyclaw_checkout "$INSTALL_DIR"; then
    info "Found an existing checkout at '$INSTALL_DIR' -- skipping clone."
elif [ -d "$INSTALL_DIR" ]; then
    error "'$INSTALL_DIR' already exists but doesn't look like a PyClaw checkout."
    error "Remove it, choose a different directory, or cd into it first if it IS PyClaw."
    exit 1
else
    info "Cloning PyClaw into '$INSTALL_DIR'..."
    git clone "$REPO_URL" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"

# ----------------------------------------------------------------------
# Virtual environment
# ----------------------------------------------------------------------
if [ -d ".venv" ]; then
    info "Virtual environment already exists at .venv -- reusing it."
else
    info "Creating virtual environment in .venv..."
    "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

info "Upgrading pip..."
pip install --upgrade pip --quiet

info "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

deactivate

# ----------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------
echo
info "PyClaw is installed in: $(pwd)"
echo
echo "  To start PyClaw:"
if [ "$INSTALL_DIR" != "." ]; then
    echo "    cd $INSTALL_DIR"
fi
echo "    source .venv/bin/activate"
echo "    python main.py"
echo
echo "  Make sure a model server (llama.cpp, Ollama, LM Studio, ...) is running first --"
echo "  see README.md for backend setup. Without one, PyClaw will start but show 'Offline'."
