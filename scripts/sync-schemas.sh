#!/usr/bin/env bash
# Sync schemas from the central schemas repository
# Usage: ./scripts/sync-schemas.sh [--dry-run]

set -euo pipefail

SCHEMA_REPO_URL="https://raw.githubusercontent.com/byteowlz/schemas/main"
SCHEMA_DIR="examples/schemas"
DRY_RUN=false

# Parse arguments
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "DRY RUN MODE - No files will be modified"
fi

# All extraction schemas
SCHEMAS=(
    "extraction/contact_details.json"
    "extraction/contract.json"
    "extraction/customer_feedback.json"
    "extraction/email_triage.json"
    "extraction/event.json"
    "extraction/invoice.json"
    "extraction/meeting_notes.json"
    "extraction/movie.json"
    "extraction/org.json"
    "extraction/resume.json"
)

mkdir -p "$SCHEMA_DIR"

echo "Syncing schemas from ${SCHEMA_REPO_URL}..."
echo

for schema in "${SCHEMAS[@]}"; do
    filename=$(basename "$schema")
    url="$SCHEMA_REPO_URL/$schema"
    dest="$SCHEMA_DIR/$filename"
    
    if [ "$DRY_RUN" = true ]; then
        echo "Would fetch: $url -> $dest"
    else
        echo "Fetching: $filename"
        if curl -sf "$url" -o "$dest"; then
            echo "  ✓ $filename synced"
        else
            echo "  ✗ Failed to fetch $schema"
            exit 1
        fi
    fi
done

echo
echo "Schema sync complete! Synced ${#SCHEMAS[@]} schemas."
