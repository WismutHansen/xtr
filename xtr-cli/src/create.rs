use anyhow::{Context, Result};
use json_schema_generator::generate_json_schema;
use std::fs;
use std::path::Path;

pub fn handle_create_schema(input_path: &str, name: Option<&str>) -> Result<()> {
    let input_path = Path::new(input_path);

    if !input_path.exists() {
        anyhow::bail!("Input file does not exist: {}", input_path.display());
    }

    let json_content = fs::read_to_string(input_path)
        .with_context(|| format!("failed to read file: {}", input_path.display()))?;

    let json_value: serde_json::Value = serde_json::from_str(&json_content)
        .with_context(|| format!("failed to parse JSON from: {}", input_path.display()))?;

    let schema = generate_json_schema(&json_value);

    let schema_name = name.unwrap_or_else(|| {
        input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("schema")
    });

    let output_filename = format!("{}.json", schema_name);
    let output_path = Path::new("examples/schemas").join(&output_filename);

    fs::create_dir_all("examples/schemas")
        .context("failed to create examples/schemas directory")?;

    let schema_json =
        serde_json::to_string_pretty(&schema).context("failed to serialize schema to JSON")?;

    fs::write(&output_path, schema_json)
        .with_context(|| format!("failed to write schema to: {}", output_path.display()))?;

    println!("Schema generated successfully:");
    println!("  Input: {}", input_path.display());
    println!("  Output: {}", output_path.display());
    println!("  Name: {}", schema_name);

    Ok(())
}
