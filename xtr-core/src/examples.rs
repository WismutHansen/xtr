use std::fs;
use std::io;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use serde::Deserialize;

/// Representation of a single training example loaded from a `.json` file.
#[derive(Debug, Clone)]
pub struct TaskExample {
    pub name: String,
    pub input_text: String,
    pub expected_json: serde_json::Value,
    pub additional_context: Option<String>,
    pub image_paths: Vec<PathBuf>,
}

impl TaskExample {
    pub fn to_training_example(&self, schema: &str, include_timestamp: bool) -> dspy_rs::Example {
        let expected_str = serde_json::to_string(&self.expected_json).unwrap_or_default();

        let mut context_parts = Vec::new();
        if let Some(ctx) = &self.additional_context {
            if !ctx.trim().is_empty() {
                context_parts.push(ctx.clone());
            }
        }

        if include_timestamp {
            let now = chrono::Local::now();
            let timestamp = now.format("%A, %B %d, %Y at %I:%M %p %Z").to_string();
            context_parts.push(format!("Current date and time: {timestamp}"));
        }

        let final_context = context_parts.join("\n\n");

        dspy_rs::example! {
            "schema": "input" => schema,
            "input_text": "input" => &self.input_text,
            "additional_context": "input" => final_context.as_str(),
            "expected_output": "input" => expected_str,
            "example_name": "input" => &self.name
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawTaskExample {
    input_text: String,
    expected_json: serde_json::Value,
    #[serde(default)]
    additional_context: Option<String>,
    #[serde(default)]
    images: Vec<String>,
}

pub fn load_task_examples(dir: &Path) -> Result<Vec<TaskExample>> {
    let mut entries = Vec::new();
    if !dir.exists() {
        return Err(anyhow::anyhow!(
            "examples directory '{}' does not exist",
            dir.display()
        ));
    }

    for entry in fs::read_dir(dir)
        .with_context(|| format!("unable to read examples directory {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }

        let content = fs::read_to_string(&path)
            .with_context(|| format!("failed to read example file '{}'", path.display()))?;

        let RawTaskExample {
            input_text,
            expected_json,
            additional_context,
            images,
        } = serde_json::from_str(&content)
            .with_context(|| format!("failed to parse example json '{}'", path.display()))?;

        let mut image_paths = Vec::with_capacity(images.len());
        for relative in &images {
            let resolved = resolve_example_path(&path, relative)?;
            image_paths.push(resolved);
        }

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("example")
            .to_string();

        entries.push(TaskExample {
            name,
            input_text,
            expected_json,
            additional_context,
            image_paths,
        });
    }

    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(entries)
}

fn resolve_example_path(example_file: &Path, relative: &str) -> Result<PathBuf> {
    let base = example_file
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "example file has no parent"))?;
    let expanded = shellexpand::full_with_context(
        relative,
        || None::<&str>,
        |var| Ok::<Option<String>, std::convert::Infallible>(std::env::var(var).ok()),
    )
    .with_context(|| format!("failed to expand path '{relative}'"))?;
    let candidate = PathBuf::from(expanded.as_ref());
    Ok(if candidate.is_absolute() {
        candidate
    } else {
        base.join(candidate)
    })
}
