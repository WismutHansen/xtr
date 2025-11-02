use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use dspy_rs::Module;
use dspy_rs::core::Optimizable as OptimizableTrait;
use dspy_rs::example;
use jsonschema::JSONSchema;
use serde_json::Value;

use crate::config::AppConfig;
use crate::config::AppPaths;
use crate::config::ConfigBundle;
use crate::config::OptimizationSettings;
use crate::config::ResolvedTaskConfig;
use crate::config::load_or_initialize_config;
use crate::lm::ModelHandle;
use crate::lm::TaskModelHandles;
use crate::optimization::ExtractionProgram;
use crate::optimization::GepaOutcome;
use crate::optimization::GepaRunner;
use crate::optimization::load_best_instruction;

/// Schema validation behavior during inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationMode {
    /// No validation performed
    None,
    /// Validate and show warning if validation fails, but return the output
    Warn,
    /// Validate and return error if validation fails
    Error,
}

/// Retry behavior when validation fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryMode {
    /// No chat history - each attempt is independent
    Independent,
    /// Include chat history and error message - LLM learns from previous attempts
    WithHistory,
}

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
        self.optimize_task_with_overrides(task_name, None).await
    }

    pub async fn optimize_task_with_overrides(
        &self,
        task_name: &str,
        overrides: Option<OptimizationSettings>,
    ) -> Result<GepaOutcome> {
        let mut task = self.resolve_task(task_name)?;

        if let Some(override_settings) = overrides {
            let task_overrides = self.bundle.config.optimization.tasks.get(task_name);
            task.optimization = self
                .bundle
                .config
                .optimization
                .resolve(Some(&override_settings), task_name)?;

            if let Some(existing_overrides) = task_overrides {
                let merged = crate::config::merge_optimization_settings_public(
                    existing_overrides,
                    Some(&override_settings),
                );
                task.optimization = self
                    .bundle
                    .config
                    .optimization
                    .resolve(Some(&merged), task_name)?;
            }
        }

        let models = TaskModelHandles::load(&task.models).await?;
        let optimization = task.optimization.clone();

        let log_dir = self.bundle.config.mlflow.log_dir.as_ref().map(|s| {
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
        self.run_inference_with_validation(
            task_name,
            input_text,
            additional_context,
            verbose,
            ValidationMode::None,
            0,
            RetryMode::Independent,
        )
        .await
    }

    pub async fn run_inference_with_validation(
        &self,
        task_name: &str,
        input_text: &str,
        additional_context: Option<&str>,
        verbose: bool,
        validation_mode: ValidationMode,
        max_retries: u32,
        retry_mode: RetryMode,
    ) -> Result<String> {
        self.run_inference_with_options(
            task_name,
            input_text,
            additional_context,
            verbose,
            validation_mode,
            max_retries,
            retry_mode,
            None,
        )
        .await
    }

    pub async fn run_inference_with_options(
        &self,
        task_name: &str,
        input_text: &str,
        additional_context: Option<&str>,
        verbose: bool,
        validation_mode: ValidationMode,
        max_retries: u32,
        retry_mode: RetryMode,
        max_tokens_override: Option<u32>,
    ) -> Result<String> {
        let mut task = self.resolve_task(task_name)?;

        // Override max_tokens if specified
        if let Some(max_tokens) = max_tokens_override {
            task.models.student.max_tokens = Some(max_tokens);
            task.models.teacher.max_tokens = Some(max_tokens);
            for fallback in &mut task.models.fallbacks {
                fallback.max_tokens = Some(max_tokens);
            }
        }

        let models = TaskModelHandles::load(&task.models).await?;

        let schema_path = task
            .schema_path
            .as_ref()
            .ok_or_else(|| anyhow!("task '{task_name}' is missing schema path"))?;
        let schema = std::fs::read_to_string(schema_path).with_context(|| {
            format!(
                "failed to read schema for task '{}' at {}",
                task_name,
                schema_path.display()
            )
        })?;

        // Parse and compile schema if validation is enabled
        let compiled_schema = if validation_mode != ValidationMode::None {
            let schema_value: Value = serde_json::from_str(&schema)
                .with_context(|| format!("failed to parse schema for task '{task_name}'"))?;

            // JSONSchema requires 'static lifetime, so we Box::leak the value
            let leaked_schema: &'static Value = Box::leak(Box::new(schema_value));
            let compiled = JSONSchema::compile(leaked_schema)
                .with_context(|| format!("failed to compile schema for task '{task_name}'"))?;

            Some(compiled)
        } else {
            None
        };

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

            // Try with retries if enabled
            let result = match retry_mode {
                RetryMode::Independent => {
                    self.try_inference_independent(
                        &mut program,
                        &schema,
                        input_text,
                        &final_context,
                        compiled_schema.as_ref(),
                        validation_mode,
                        max_retries,
                        verbose,
                    )
                    .await
                }
                RetryMode::WithHistory => {
                    self.try_inference_with_history(
                        &mut program,
                        &schema,
                        input_text,
                        &final_context,
                        compiled_schema.as_ref(),
                        validation_mode,
                        max_retries,
                        verbose,
                    )
                    .await
                }
            };

            match result {
                Ok(output) => return Ok(output),
                Err(err) => {
                    errors.push(format!("{label}: {err}"));
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
            "all configured models failed to produce valid output: {summary}"
        ))
    }

    async fn try_inference_independent(
        &self,
        program: &mut ExtractionProgram,
        schema: &str,
        input_text: &str,
        additional_context: &str,
        compiled_schema: Option<&JSONSchema>,
        validation_mode: ValidationMode,
        max_retries: u32,
        verbose: bool,
    ) -> Result<String> {
        let attempts = max_retries + 1;

        for attempt in 0..attempts {
            let example = example! {
                "schema": "input" => schema,
                "input_text": "input" => input_text,
                "additional_context": "input" => additional_context
            };

            let prediction = program.forward(example).await.context("inference failed")?;

            let output = Self::extract_output_json(&prediction)?;

            // Validate if schema validation is enabled
            if let Some(compiled) = compiled_schema {
                match Self::validate_with_compiled(&output, compiled) {
                    Ok(_) => return Ok(output),
                    Err(validation_err) => match validation_mode {
                        ValidationMode::None => return Ok(output),
                        ValidationMode::Warn => {
                            eprintln!("Warning: Schema validation failed: {validation_err}");
                            return Ok(output);
                        }
                        ValidationMode::Error => {
                            if attempt < max_retries {
                                if verbose {
                                    eprintln!(
                                        "Attempt {}: validation failed, retrying... ({validation_err})",
                                        attempt + 1
                                    );
                                }
                                continue;
                            } else {
                                return Err(anyhow!(
                                    "schema validation failed after {attempts} attempts: {validation_err}"
                                ));
                            }
                        }
                    },
                }
            } else {
                return Ok(output);
            }
        }

        Err(anyhow!(
            "failed to produce valid output after {attempts} attempts"
        ))
    }

    async fn try_inference_with_history(
        &self,
        program: &mut ExtractionProgram,
        schema: &str,
        input_text: &str,
        additional_context: &str,
        compiled_schema: Option<&JSONSchema>,
        validation_mode: ValidationMode,
        max_retries: u32,
        verbose: bool,
    ) -> Result<String> {
        let attempts = max_retries + 1;
        let mut context_with_feedback = additional_context.to_string();

        for attempt in 0..attempts {
            let example = example! {
                "schema": "input" => schema,
                "input_text": "input" => input_text,
                "additional_context": "input" => context_with_feedback.as_str()
            };

            let prediction = program.forward(example).await.context("inference failed")?;

            let output = Self::extract_output_json(&prediction)?;

            // Validate if schema validation is enabled
            if let Some(compiled) = compiled_schema {
                match Self::validate_with_compiled(&output, compiled) {
                    Ok(_) => return Ok(output),
                    Err(validation_err) => {
                        match validation_mode {
                            ValidationMode::None => return Ok(output),
                            ValidationMode::Warn => {
                                eprintln!("Warning: Schema validation failed: {validation_err}");
                                return Ok(output);
                            }
                            ValidationMode::Error => {
                                if attempt < max_retries {
                                    if verbose {
                                        eprintln!(
                                            "Attempt {}: validation failed, retrying with feedback... ({validation_err})",
                                            attempt + 1
                                        );
                                    }
                                    // Add error feedback to context for next attempt
                                    context_with_feedback = format!(
                                        "{additional_context}\n\nPrevious attempt failed validation with error: {validation_err}\nPlease correct the output to match the schema."
                                    );
                                    continue;
                                } else {
                                    return Err(anyhow!(
                                        "schema validation failed after {attempts} attempts: {validation_err}"
                                    ));
                                }
                            }
                        }
                    }
                }
            } else {
                return Ok(output);
            }
        }

        Err(anyhow!(
            "failed to produce valid output after {attempts} attempts"
        ))
    }

    fn validate_with_compiled(output: &str, compiled_schema: &JSONSchema) -> Result<()> {
        let json: Value = serde_json::from_str(output).context("output is not valid JSON")?;

        if let Err(errors) = compiled_schema.validate(&json) {
            let error_messages: Vec<String> = errors.map(|e| format!("{e}")).collect();
            return Err(anyhow!("validation errors: {}", error_messages.join("; ")));
        }

        Ok(())
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
