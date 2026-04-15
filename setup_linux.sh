#!/usr/bin/env bash
# batteryplot — Linux/macOS setup script
# Run once from the repo root:  bash setup_linux.sh
# Then activate and use:        source .venv/bin/activate && batteryplot run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== batteryplot setup ==="

# --- Python version check ---
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(sys.version_info[:2])")
        # Accept 3.11+
        ok=$("$candidate" -c "import sys; print(int(sys.version_info >= (3,11)))")
        if [[ "$ok" == "1" ]]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "ERROR: Python 3.11 or later not found on PATH."
    echo "Install it via your package manager, e.g.:"
    echo "  Arch:   sudo pacman -S python"
    echo "  Ubuntu: sudo apt install python3.11"
    echo "  Fedora: sudo dnf install python3.11"
    exit 1
fi

echo "Using Python: $($PYTHON --version)"

# --- Virtual environment ---
VENV_DIR="$SCRIPT_DIR/.venv"
if [[ ! -f "$VENV_DIR/bin/python" ]]; then
    echo "Creating virtual environment at .venv ..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists at .venv"
fi

# --- Install package (entry point + dependencies) ---
echo "Installing batteryplot and dependencies ..."
"$VENV_DIR/bin/pip" install --upgrade pip --quiet
# pip install -e . reads pyproject.toml, installs all deps, and registers
# the 'batteryplot' console script entry point in .venv/bin/
"$VENV_DIR/bin/pip" install -e "$SCRIPT_DIR" --quiet

# --- Write default config if not present ---
if [[ ! -f "$SCRIPT_DIR/config.yaml" ]]; then
    echo "Writing default config.yaml ..."
    "$VENV_DIR/bin/batteryplot" init-config
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  # Drop your CSV files into:  $(realpath input)"
echo "  batteryplot run"
echo ""
echo "Other useful commands:"
echo "  batteryplot inspect path/to/file.csv   # parse and report column mapping"
echo "  batteryplot validate                    # check all CSVs in input/"
echo "  batteryplot list-plots                  # show all 13 plot types"
echo "  batteryplot --help"
