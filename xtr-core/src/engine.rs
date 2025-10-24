use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use dspy_rs::Module;
use dspy_rs::core::Optimizable as OptimizableTrait;
use dspy_rs::example;

use crate::config::AppConfig;
use crate::config::AppPaths;
use crate::config::ConfigBundle;
use crate::config::ResolvedTaskConfig;
use crate::config::load_or_initialize_config;
use crate::lm::ModelHandle;
use crate::lm::TaskModelHandles;
use crate::optimization::ExtractionProgram;
use crate::optimization::GepaOutcome;
use crate::optimization::GepaRunner;
use crate::optimization::load_best_instruction;

/// High-level orchestrator for optimization and inference workflows.
#[derive(Debug)]
pub struct ExtractionEngine {
    bundle: ConfigBundle,
}

impl ExtractionEngine {
    /// Load configuration from disk (creating defaults if needed) and produce a
    /// ready-to-use engine instance.
    pub fn load(app_name: impl AsRef<str>) -> Result<Self> {
        Ok(Self {
            bundle: load_or_initialize_config(app_name)?,
        })
    }

    /// Construct an engine from an existing [`ConfigBundle`]. Useful for tests.
    pub fn from_bundle(bundle: ConfigBundle) -> Self {
        Self { bundle }
    }

    pub fn config(&self) -> &AppConfig {
        &self.bundle.config
    }

    pub fn paths(&self) -> &AppPaths {
        &self.bundle.paths
    }

    pub fn resolve_task(&self, task_name: &str) -> Result<ResolvedTaskConfig> {
        self.bundle
            .config
            .resolve_task(task_name, &self.bundle.paths)
    }

    pub async fn task_models(&self, task_name: &str) -> Result<TaskModelHandles> {
        let task = self.resolve_task(task_name)?;
        TaskModelHandles::load(&task.models).await
    }

    pub async fn optimize_task(&self, task_name: &str) -> Result<GepaOutcome> {
        let task = self.resolve_task(task_name)?;
        let models = TaskModelHandles::load(&task.models).await?;
        let optimization = task.optimization.clone();

        let log_dir = self.bundle.config.mlflow.log_dir.as_ref()
            .map(|s| {
                let expanded = shellexpand::tilde(s);
                std::path::PathBuf::from(expanded.as_ref())
            });

        let runner = GepaRunner {
            task,
            optimization,
            models,
            #[cfg(feature = "mlflow")]
            mlflow_tracking_uri: self.bundle.config.mlflow.tracking_uri.clone(),
            #[cfg(feature = "mlflow")]
            mlflow_experiment: self.bundle.config.mlflow.experiment_name.clone(),
            local_logging: self.bundle.config.mlflow.local_logging,
            log_dir,
        };

        runner.run(&self.bundle.paths.state_dir).await
    }

    pub async fn run_inference(
        &self,
        task_name: &str,
        input_text: &str,
        additional_context: Option<&str>,
        verbose: bool,
    ) -> Result<String> {
        let task = self.resolve_task(task_name)?;
        let models = TaskModelHandles::load(&task.models).await?;

        let schema_path = task
            .schema_path
            .as_ref()
            .ok_or_else(|| anyhow!("task '{}' is missing schema path", task_name))?;
        let schema = std::fs::read_to_string(schema_path).with_context(|| {
            format!(
                "failed to read schema for task '{}' at {}",
                task_name,
                schema_path.display()
            )
        })?;

        let mut program = ExtractionProgram::with_verbose(verbose);

        if let Some(best) = load_best_instruction(&self.bundle.paths, task_name) {
            OptimizableTrait::update_signature_instruction(program.solver_mut(), best)?;
        } else if let Some(description) = &task.description {
            let base = OptimizableTrait::get_signature(program.solver()).instruction();
            let combined = if base.trim().is_empty() {
                description.clone()
            } else {
                format!("{base}\n\nTask context: {description}")
            };
            OptimizableTrait::update_signature_instruction(program.solver_mut(), combined)?;
        }

        let mut context_parts = Vec::new();
        if let Some(ctx) = additional_context {
            if !ctx.trim().is_empty() {
                context_parts.push(ctx.to_string());
            }
        }
        
        if task.include_timestamp {
            let now = chrono::Local::now();
            let timestamp = now.format("%A, %B %d, %Y at %I:%M %p %Z").to_string();
            context_parts.push(format!("Current date and time: {timestamp}"));
        }
        
        let final_context = context_parts.join("\n\n");

        let example = example! {
            "schema": "input" => &schema,
            "input_text": "input" => input_text,
            "additional_context": "input" => final_context.as_str()
        };

        let mut attempts: Vec<(&ModelHandle, String)> =
            Vec::with_capacity(2 + models.fallbacks.len());
        attempts.push((
            &models.student,
            format!("student '{}'", models.student.descriptor.name),
        ));
        attempts.push((
            &models.teacher,
            format!("teacher '{}'", models.teacher.descriptor.name),
        ));
        for fallback in &models.fallbacks {
            attempts.push((fallback, format!("fallback '{}'", fallback.descriptor.name)));
        }

        let mut errors = Vec::new();
        for (handle, label) in attempts {
            handle.configure_global();
            let prediction = match program.forward(example.clone()).await {
                Ok(prediction) => prediction,
                Err(err) => {
                    errors.push(format!("{label} inference error: {err}"));
                    continue;
                }
            };

            match Self::extract_output_json(&prediction) {
                Ok(output) => return Ok(output),
                Err(err) => {
                    errors.push(format!("{label} returned invalid output: {err}"));
                    continue;
                }
            }
        }

        let summary = if errors.is_empty() {
            "no models were available to run inference".to_string()
        } else {
            errors.join("; ")
        };

        Err(anyhow!(
            "all configured models failed to produce a usable 'output_json': {summary}"
        ))
    }

    fn extract_output_json(prediction: &dspy_rs::Prediction) -> Result<String> {
        let raw_value = prediction.get("output_json", None);

        fn strip_code_fence(text: &str) -> Option<String> {
            let trimmed = text.trim();
            if !trimmed.starts_with("```") {
                return None;
            }
            let mut parts = trimmed.splitn(2, '\n');
            parts.next()?;
            let remainder = parts.next()?.trim();
            let end = remainder.rfind("```")?;
            Some(remainder[..end].trim().to_string())
        }

        let make_object_string = |value: serde_json::Value| -> Result<String> {
            match value {
                serde_json::Value::String(inner) => {
                    if inner.trim().is_empty() {
                        Err(anyhow!("response contained an empty output_json string"))
                    } else {
                        Ok(inner)
                    }
                }
                serde_json::Value::Object(_) | serde_json::Value::Array(_) => Ok(value.to_string()),
                other => {
                    let rendered = other.to_string();
                    if rendered.trim().is_empty() || rendered == "null" {
                        Err(anyhow!("response contained empty output_json value"))
                    } else {
                        Ok(rendered)
                    }
                }
            }
        };

        match raw_value {
            serde_json::Value::Null => Err(anyhow!("model response missing 'output_json' field")),
            serde_json::Value::String(s) => {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    return Err(anyhow!(
                        "model response returned empty 'output_json' string"
                    ));
                }

                let candidate = strip_code_fence(trimmed).unwrap_or_else(|| trimmed.to_string());
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&candidate) {
                    if let serde_json::Value::String(inner) = parsed {
                        if inner.trim().is_empty() {
                            Err(anyhow!(
                                "model response encoded an empty JSON string for 'output_json'"
                            ))
                        } else {
                            Ok(inner)
                        }
                    } else if matches!(
                        parsed,
                        serde_json::Value::Object(_) | serde_json::Value::Array(_)
                    ) {
                        Ok(parsed.to_string())
                    } else {
                        make_object_string(parsed)
                    }
                } else {
                    Ok(candidate)
                }
            }
            other @ serde_json::Value::Object(_) | other @ serde_json::Value::Array(_) => {
                make_object_string(other)
            }
            other => make_object_string(other),
        }
    }
}
