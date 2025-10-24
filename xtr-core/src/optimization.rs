use std::fs;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use bon::Builder;
use dspy_rs::Evaluator;
use dspy_rs::Module;
use dspy_rs::Optimizable;
use dspy_rs::Predict;
use dspy_rs::Prediction;
use dspy_rs::Predictor;
use dspy_rs::Signature;
use dspy_rs::core::Optimizable as OptimizableTrait;
use dspy_rs::evaluate::FeedbackEvaluator;
use dspy_rs::evaluate::FeedbackMetric;
use dspy_rs::optimizer::GEPA;
use dspy_rs::optimizer::GEPAResult;
use jsonschema::JSONSchema;
use serde_json::Value;

use crate::config::AppPaths;
use crate::config::ResolvedOptimizationSettings;
use crate::config::ResolvedTaskConfig;
use crate::examples::load_task_examples;
use crate::lm::TaskModelHandles;
use crate::lm::build_model_handle;

#[Signature(cot)]
pub struct ExtractionSignature {
    /// You convert unstructured inputs into structured JSON. Only include fields that are present in the input. Omit fields that have no value.

    #[input(desc = "JSON schema describing the expected output object")]
    pub schema: String,

    #[input(desc = "Primary text content to extract data from")]
    pub input_text: String,

    #[input(desc = "Additional context, metadata, or task-specific guidance")]
    pub additional_context: String,

    #[output(desc = "A JSON object that strictly adheres to the provided schema")]
    pub output_json: String,
}

#[derive(Builder, Optimizable)]
pub struct ExtractionProgram {
    #[parameter]
    solver: Predict,

    #[builder(default = false)]
    verbose: bool,
}

impl ExtractionProgram {
    pub fn new() -> Self {
        Self::builder()
            .solver(Predict::new(ExtractionSignature::new()))
            .build()
    }

    pub fn with_verbose(verbose: bool) -> Self {
        Self::builder()
            .solver(Predict::new(ExtractionSignature::new()))
            .verbose(verbose)
            .build()
    }

    pub fn solver(&self) -> &Predict {
        &self.solver
    }

    pub fn solver_mut(&mut self) -> &mut Predict {
        &mut self.solver
    }
}

impl Module for ExtractionProgram {
    async fn forward(&self, inputs: dspy_rs::Example) -> Result<Prediction> {
        let prediction = self.solver.forward(inputs).await?;

        if self.verbose {
            eprintln!("DEBUG: Prediction keys: {:?}", prediction.keys());
            eprintln!("DEBUG: Prediction data: {:?}", prediction.data);
        }

        let output_json_val = prediction.get("output_json", None);

        if let Some(output_str) = output_json_val.as_str() {
            if !output_str.is_empty() {
                if let Ok(Value::String(inner)) = serde_json::from_str::<Value>(output_str) {
                    if self.verbose {
                        eprintln!("Successfully unescaped JSON string");
                    }
                    let mut fixed_prediction = Prediction::from(prediction);
                    fixed_prediction
                        .data
                        .insert("output_json".to_string(), inner.into());
                    return Ok(fixed_prediction);
                }
            }
        }

        Ok(prediction)
    }
}

impl Evaluator for ExtractionProgram {
    async fn metric(&self, example: &dspy_rs::Example, prediction: &Prediction) -> f32 {
        self.feedback_metric(example, prediction).await.score
    }
}

impl FeedbackEvaluator for ExtractionProgram {
    async fn feedback_metric(
        &self,
        example: &dspy_rs::Example,
        prediction: &Prediction,
    ) -> FeedbackMetric {
        let schema_str = example
            .get("schema", None)
            .as_str()
            .unwrap_or("")
            .to_string();

        let expected_raw = example
            .get("expected_output", None)
            .as_str()
            .unwrap_or("")
            .to_string();

        let input_text = example
            .get("input_text", None)
            .as_str()
            .unwrap_or("")
            .to_string();

        let predicted = prediction
            .get("output_json", None)
            .as_str()
            .unwrap_or("")
            .to_string();

        let predicted = if let Ok(Value::String(inner)) = serde_json::from_str::<Value>(&predicted)
        {
            inner
        } else {
            predicted
        };

        let schema_json: Value = match serde_json::from_str(&schema_str) {
            Ok(value) => value,
            Err(err) => {
                return FeedbackMetric::new(
                    0.0,
                    format!(
                        "Schema parse error: {err}. Cannot validate output for input: {input_text}"
                    ),
                );
            }
        };

        let expected_json: Value = match serde_json::from_str(&expected_raw) {
            Ok(value) => value,
            Err(err) => {
                return FeedbackMetric::new(
                    0.0,
                    format!(
                        "Expected output parse error: {err}. Review training data for input: {input_text}"
                    ),
                );
            }
        };

        let predicted_json: Value = match serde_json::from_str(&predicted) {
            Ok(value) => value,
            Err(err) => {
                return FeedbackMetric::new(
                    0.0,
                    format!("Generated output was not valid JSON ({err}). Input: {input_text}"),
                );
            }
        };

        let compiled = match JSONSchema::compile(&schema_json) {
            Ok(schema) => schema,
            Err(err) => {
                return FeedbackMetric::new(
                    0.0,
                    format!(
                        "Failed to compile schema: {err}. Ensure the schema is valid JSON Schema."
                    ),
                );
            }
        };

        let mut score: f32 = 0.0;
        let mut feedback_lines = Vec::new();

        if let Err(errors) = compiled.validate(&predicted_json) {
            feedback_lines.push("Schema validation failed:".to_string());
            for error in errors.take(5) {
                feedback_lines.push(format!("- {}", error));
            }
        } else {
            score += 0.4_f32;
            feedback_lines.push("✅ Output satisfies JSON schema.".to_string());
        }

        if predicted_json == expected_json {
            score = 1.0_f32;
            feedback_lines.push("✅ Output matches the expected JSON exactly.".to_string());
        } else {
            // Compare object keys for partial credit
            let attempted_keys = predicted_json
                .as_object()
                .map(|map| map.keys().cloned().collect::<Vec<_>>())
                .unwrap_or_default();
            let expected_keys = expected_json
                .as_object()
                .map(|map| map.keys().cloned().collect::<Vec<_>>())
                .unwrap_or_default();

            if !attempted_keys.is_empty() && !expected_keys.is_empty() {
                let overlap = attempted_keys
                    .iter()
                    .filter(|key| expected_keys.contains(key))
                    .count();
                if overlap > 0 {
                    score = score.max(0.6_f32);
                    feedback_lines.push(format!(
                        "⚠️ Output captured {overlap} / {} expected keys.",
                        expected_keys.len()
                    ));
                } else {
                    feedback_lines.push("⚠️ Output did not capture expected keys.".to_string());
                }
            }

            // Provide diff-style feedback
            feedback_lines.push(format!(
                "Expected: {}\nGenerated: {}",
                expected_raw, predicted
            ));
        }

        FeedbackMetric::new(score, feedback_lines.join("\n"))
    }
}

#[derive(Clone)]
pub struct GepaOutcome {
    pub best_instruction: String,
    pub best_score: f32,
    pub total_iterations: usize,
    pub total_rollouts: usize,
    pub total_lm_calls: usize,
}

pub struct GepaRunner {
    pub task: ResolvedTaskConfig,
    pub optimization: ResolvedOptimizationSettings,
    pub models: TaskModelHandles,
    #[cfg(feature = "mlflow")]
    pub mlflow_tracking_uri: Option<String>,
    #[cfg(feature = "mlflow")]
    pub mlflow_experiment: Option<String>,
    pub local_logging: bool,
    pub log_dir: Option<std::path::PathBuf>,
}

impl GepaRunner {
    pub async fn run(self, state_dir: &Path) -> Result<GepaOutcome> {
        let GepaRunner {
            task,
            optimization,
            models,
            #[cfg(feature = "mlflow")]
            mlflow_tracking_uri,
            #[cfg(feature = "mlflow")]
            mlflow_experiment,
            local_logging,
            log_dir,
        } = self;

        #[cfg(feature = "mlflow")]
        let logger = crate::mlflow_logger::GepaLogger::new(
            task.name.clone(),
            mlflow_tracking_uri,
            mlflow_experiment,
            local_logging,
            log_dir,
        )?;

        #[cfg(not(feature = "mlflow"))]
        let logger = crate::mlflow_logger::GepaLogger::new(
            task.name.clone(),
            local_logging,
            log_dir,
        )?;

        let schema_path = task
            .schema_path
            .as_ref()
            .ok_or_else(|| anyhow!("task '{}' is missing schema path", task.name))?;
        let schema = fs::read_to_string(schema_path).with_context(|| {
            format!(
                "failed to read schema for task '{}' at {}",
                task.name,
                schema_path.display()
            )
        })?;

        let examples_dir = task
            .examples_dir
            .as_ref()
            .ok_or_else(|| anyhow!("task '{}' is missing examples directory", task.name))?;

        let dataset = load_task_examples(examples_dir)?;
        if dataset.is_empty() {
            return Err(anyhow!(
                "no training examples found in {}",
                examples_dir.display()
            ));
        }

        models.student.configure_global();

        let mut program = ExtractionProgram::new();
        if let Some(description) = &task.description {
            let base = OptimizableTrait::get_signature(&program.solver).instruction();
            let combined = if base.trim().is_empty() {
                description.clone()
            } else {
                format!("{base}\n\nTask context: {description}")
            };
            OptimizableTrait::update_signature_instruction(&mut program.solver, combined)?;
        }

        let train_examples: Vec<_> = dataset
            .iter()
            .map(|example| example.to_training_example(&schema, task.include_timestamp))
            .collect();

        let prompt_model = if let Some(descriptor) = optimization.feedback_models.teacher.clone() {
            Some(build_model_handle(&descriptor).await?.lm)
        } else {
            Some(models.teacher.lm.clone())
        };

        let gepa = GEPA {
            num_iterations: usize::max(1, optimization.iterations as usize),
            minibatch_size: usize::max(1, optimization.batch_size as usize),
            num_trials: usize::max(1, optimization.rollouts_per_iteration as usize),
            temperature: 0.9,
            track_stats: true,
            track_best_outputs: false,
            max_rollouts: None,
            max_lm_calls: (optimization.max_lm_calls > 0)
                .then_some(optimization.max_lm_calls as usize),
            prompt_model,
            valset: None,
        };
        let config = serde_json::json!({
            "task_name": task.name,
            "num_iterations": optimization.iterations,
            "minibatch_size": optimization.batch_size,
            "rollouts_per_iteration": optimization.rollouts_per_iteration,
            "max_lm_calls": optimization.max_lm_calls,
            "student_model": models.student.descriptor.name,
            "teacher_model": models.teacher.descriptor.name,
        });
        logger.log_config(&config)?;

        let result = gepa
            .compile_with_feedback(&mut program, train_examples)
            .await?;

        persist_gepa_result(state_dir, &task.name, &result)?;

        logger.log_final_result(&result)?;
        logger.finish()?;

        Ok(GepaOutcome {
            best_instruction: result.best_candidate.instruction.clone(),
            best_score: result.best_candidate.average_score(),
            total_iterations: result.evolution_history.len(),
            total_rollouts: result.total_rollouts,
            total_lm_calls: result.total_lm_calls,
        })
    }
}

fn persist_gepa_result(state_dir: &Path, task_name: &str, result: &GEPAResult) -> Result<()> {
    let task_dir = state_dir.join("prompts").join(task_name);
    fs::create_dir_all(&task_dir)?;

    fs::write(
        task_dir.join("best.meta"),
        format!(
            "best_score={:.4}\ntotal_rollouts={}\ntotal_lm_calls={}",
            result.best_candidate.average_score(),
            result.total_rollouts,
            result.total_lm_calls
        ),
    )?;
    fs::write(
        task_dir.join("best.txt"),
        &result.best_candidate.instruction,
    )?;

    let serialized = serde_json::to_vec_pretty(result)?;
    fs::write(task_dir.join("result.json"), serialized)?;

    Ok(())
}

pub fn load_best_instruction(paths: &AppPaths, task_name: &str) -> Option<String> {
    let best_path = paths
        .state_dir
        .join("prompts")
        .join(task_name)
        .join("best.txt");
    fs::read_to_string(best_path).ok()
}
