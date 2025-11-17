#!/usr/bin/env bash
# Validate examples against schemas (Rust version)
# Usage: ./scripts/validate-schemas.sh

set -euo pipefail

SCHEMA_DIR="examples/schemas"
EXAMPLES_DIR="examples/examples"

# Check if jsonschema is installed
if ! command -v jsonschema &> /dev/null; then
    echo "Error: jsonschema CLI not found. Install with:"
    echo "  pip install jsonschema"
    echo "  or: uv tool install jsonschema"
    exit 1
fi

echo "Validating examples against schemas..."
echo

total_validated=0
total_errors=0

for schema_file in "$SCHEMA_DIR"/*.json; do
    schema_name=$(basename "$schema_file" .json")
    example_dir="$EXAMPLES_DIR/$schema_name"
    
    if [ -d "$example_dir" ]; then
        echo "Validating $schema_name examples..."
        example_count=0
        error_count=0
        
        for example in "$example_dir"/*.json; do
            if [ -f "$example" ]; then
                example_basename=$(basename "$example")
                if jsonschema -i "$example" "$schema_file" 2>&1 >/dev/null; then
                    echo "  ✓ $example_basename"
                    ((example_count++))
                else
                    echo "  ✗ $example_basename FAILED"
                    ((error_count++))
                fi
            fi
        done
        
        if [ "$error_count" -gt 0 ]; then
            echo "  ⚠ $error_count validation errors in $schema_name"
            ((total_errors+=error_count))
        fi
        
        ((total_validated+=example_count))
    fi
done

echo
if [ "$total_errors" -gt 0 ]; then
    echo "Validation complete: $total_validated passed, $total_errors failed"
    exit 1
else
    echo "All validations passed! ($total_validated examples validated)"
fi
