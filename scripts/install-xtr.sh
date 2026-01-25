#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Installing xtr CLI..."
cargo install --path "${ROOT_DIR}/xtr-cli" --force

echo ""
echo "Installation complete!"
echo "Binary is available in \$HOME/.cargo/bin"
echo ""
echo "Quick start:"
echo "  xtr --help              # Show help"
echo "  xtr extract <schema>    # Extract data using schema"
