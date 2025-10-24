# XTR

Structured data extraction engine powered by GEPA (Guided Evolution of Prompts and Adapters).

## Overview

XTR extracts structured data from unstructured text using language models and automatically optimizes extraction quality through iterative prompt refinement.

## Features

- **Schema-driven extraction**: Define JSON schemas for your data structures
- **Multi-model fallback**: Teacher/student model architecture with fallback support
- **Auto-optimization**: GEPA algorithm improves extraction quality over time
- **Local & cloud models**: Works with OpenAI, Anthropic, local LLMs (LM Studio, Ollama, etc.)
- **MLflow integration**: Track optimization experiments and results

## Quick Start

```bash
# Extract contact details
echo "Contact John at john@example.com" | xtr infer contact_details

# Extract event information
cat event_announcement.txt | xtr infer event

# Optimize extraction for a task
xtr optimize contact_details
```

## Configuration

Configuration is stored in `~/.config/xtr/config.toml` (or `$XDG_CONFIG_HOME/xtr/config.toml`).

### Model Configuration

```toml
[models.defaults.teacher]
name = "gpt-4o-mini"
base_url = "https://api.openai.com/v1"
api_key = "sk-..."
adapter = "chat"

[models.defaults.student]
name = "qwen3-coder-30b"
base_url = "http://localhost:1234/v1"
api_key = "not_needed"
adapter = "chat"
```

### Using Local Models (LM Studio, Ollama)

When using local model servers with explicit `base_url`, **avoid provider prefixes** in model names (e.g., use `qwen3-coder-30b` instead of `qwen/qwen3-coder-30b`). The underlying dspy-rs library detects provider prefixes (`openai/`, `anthropic/`, `qwen/`, etc.) and automatically rewrites the `base_url` to the cloud service endpoint, ignoring your explicit configuration.

**Workaround for LM Studio models with slashes:**

LM Studio requires the exact model ID including the slash (e.g., `qwen/qwen3-coder-30b`). Since dspy-rs will override your `base_url`, you have two options:

1. **Use model alias/renaming** in LM Studio (if supported by your version)
2. **Use Ollama instead**, which doesn't require slashes in model names

Example for LM Studio (use model name without prefix if possible):

```toml
[models.defaults.student]
name = "qwen3-coder-30b"  # Without qwen/ prefix
base_url = "http://localhost:1234/v1"
api_key = "not_needed"
adapter = "chat"
```

Example for Ollama (works perfectly):

```toml
[models.defaults.student]
name = "llama3.2:3b"  # No provider prefix needed
base_url = "http://localhost:11434/v1"
api_key = "not_needed"
adapter = "chat"
```

**Note**: This is a limitation of the dspy-rs dependency. Weare tracking this issue and will provide a proper fix in a future release.

### Context Length Considerations

Large inputs (HTML pages, long documents) may exceed your model's context window. If you encounter errors like "Incorrect API key provided" when processing large files, this is often a **context overflow** error being misinterpreted.

**Solutions:**

1. **Increase context in LM Studio**:
   - Open LM Studio → Click the loaded model
   - Find "Context Length" slider
   - Increase to 32768 or higher
   - Reload the model

2. **Preprocess input** - Extract relevant text before piping to xtr:

```bash
# Extract only relevant sections
curl -s https://example.com/contact | grep -A 100 "contact" | head -500 | xtr infer contact_details

# Convert HTML to text first (requires pandoc)
curl -s https://example.com/page.html | pandoc -f html -t plain | xtr infer contact_details

# Use pup to extract specific HTML elements
curl -s https://example.com | pup '.contact-info text{}' | xtr infer contact_details
```

3. **Use a model with larger context**: Switch to models supporting 32K+ tokens (e.g., Qwen2.5, Llama 3.1)

## Task Configuration

Define tasks in your config:

```toml
[tasks."contact_details"]
schema = "$XDG_DATA_HOME/xtr/schemas/contact_details.json"
examples = "$XDG_DATA_HOME/xtr/examples/contact_details"
description = "Extract contact information from the input."

[tasks."event"]
schema = "$XDG_DATA_HOME/xtr/schemas/event.json"
examples = "$XDG_DATA_HOME/xtr/examples/event"
description = "Extract event details from the input."
include_timestamp = true
```

## Optimization

Run GEPA optimization to improve extraction:

```bash
xtr optimize contact_details
```

Optimization settings:

```toml
[optimization.defaults]
iterations = 4
rollouts_per_iteration = 6
max_lm_calls = 32
batch_size = 4

[optimization.tasks."contact_details"]
iterations = 6
rollouts_per_iteration = 8
```

## Storage Locations

- **Config**: `~/.config/xtr/` or `$XDG_CONFIG_HOME/xtr/`
- **Data**: `~/.local/share/xtr/` or `$XDG_DATA_HOME/xtr/`
- **State**: `~/.local/state/xtr/` or `$XDG_STATE_HOME/xtr/`
- **Logs**: `~/.local/state/xtr/optimization_logs/`

## MLflow Tracking

Enable experiment tracking:

```toml
[mlflow]
local_logging = true
log_dir = "~/.local/state/xtr/optimization_logs"
tracking_uri = "http://localhost:5000"
experiment_name = "xtr-optimization"
```

## Troubleshooting

### Error: "Incorrect API key provided" with local models

This error from local models (LM Studio, Ollama) usually indicates:
- **Context length exceeded** - The input is too large for your model's context window (see Context Length Considerations above)
- **Model server not running** - Verify with `curl http://localhost:1234/v1/models`
- **Wrong base_url in config** - Check your config.toml has the correct endpoint

### Error: "model response returned empty 'output_json' string"

The model returned a response but didn't follow the expected JSON format. Try:
- Optimizing the task: `xtr optimize contact_details`
- Using a different or larger model
- Adding more examples to your task's examples directory

### Model names with provider prefixes (e.g., `qwen/`, `anthropic/`)

When using **cloud APIs** (OpenAI, Anthropic, etc.), provider prefixes in model names work correctly:
- `openai/gpt-4o` → routes to `https://api.openai.com/v1`
- `anthropic/claude-3.5-sonnet` → routes to `https://api.anthropic.com/v1`
- `qwen/qwen3-coder-30b` → routes to `https://dashscope-intl.aliyuncs.com`

However, when using **local servers with explicit `base_url`**, remove provider prefixes from model names to prevent dspy-rs from overriding your endpoint configuration. Use model names without slashes: `gpt-4o`, `claude-3.5-sonnet`, `qwen3-coder-30b`, etc.

## Development

```bash
# Build
cargo build

# Run tests
cargo test

# Format code
just fmt

# Run linter
just fix -p xtr-cli
```

## License

[Add license information]
