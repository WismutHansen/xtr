use std::fs;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use bon::Builder;
use dspy_rs::Evaluator;
use dspy_rs::FeedbackEvaluator;
use dspy_rs::FeedbackMetric;
use dspy_rs::Module;
use dspy_rs::Optimizable;
use dspy_rs::Predict;
use dspy_rs::Prediction;
use dspy_rs::Predictor;
use dspy_rs::Signature;
use dspy_rs::core::Optimizable as OptimizableTrait;
use dspy_rs::optimizer::GEPA;
use dspy_rs::optimizer::GEPAResult;
use jsonschema::JSONSchema;
use serde_json::Value;

use crate::config::AppPaths;
use crate::config::MetricsConfig;
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

    #[builder(default = MetricsConfig::default())]
    metrics_config: MetricsConfig,
}

impl Default for ExtractionProgram {
    fn default() -> Self {
        Self::new()
    }
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

    pub fn with_metrics_config(metrics_config: MetricsConfig) -> Self {
        Self::builder()
            .solver(Predict::new(ExtractionSignature::new()))
            .metrics_config(metrics_config)
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

        if let Some(output_str) = output_json_val.as_str()
            && !output_str.is_empty()
            && let Ok(Value::String(inner)) = serde_json::from_str::<Value>(output_str)
        {
            if self.verbose {
                eprintln!("Successfully unescaped JSON string");
            }
            let mut fixed_prediction = prediction;
            fixed_prediction
                .data
                .insert("output_json".to_string(), inner.into());
            return Ok(fixed_prediction);
        }

        Ok(prediction)
    }
}

/// Calculate F-beta score combining precision and recall.
/// Beta > 1 favors recall, beta < 1 favors precision.
fn f_beta(precision: f32, recall: f32, beta: f32) -> f32 {
    if precision <= 0.0 && recall <= 0.0 {
        return 0.0;
    }

    let beta_sq = beta * beta;
    let denominator = (beta_sq * precision) + recall;
    if denominator <= 0.0 {
        return 0.0;
    }

    ((1.0 + beta_sq) * precision * recall) / denominator
}

impl Evaluator for ExtractionProgram {
    async fn metric(&self, example: &dspy_rs::Example, prediction: &Prediction) -> f32 {
        self.feedback_metric(example, prediction).await.score
    }
}

/// Recursively flatten a JSON value into field paths with their values.
///
/// Examples:
/// - `{"name": "John", "age": 30}` -> `{"name": "John", "age": 30}`
/// - `{"user": {"name": "John"}}` -> `{"user.name": "John"}`
/// - `{"items": [{"id": 1}, {"id": 2}]}` -> `{"items[0].id": 1, "items[1].id": 2}`
fn get_all_fields(obj: &Value, prefix: &str) -> indexmap::IndexMap<String, Value> {
    let mut fields = indexmap::IndexMap::new();

    match obj {
        Value::Object(map) => {
            for (key, value) in map {
                let full_key = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };

                if value.is_object() || value.is_array() {
                    fields.extend(get_all_fields(value, &full_key));
                } else {
                    fields.insert(full_key, value.clone());
                }
            }
        }
        Value::Array(arr) => {
            for (i, item) in arr.iter().enumerate() {
                let indexed_key = format!("{prefix}[{i}]");
                fields.extend(get_all_fields(item, &indexed_key));
            }
        }
        _ => {
            if !prefix.is_empty() {
                fields.insert(prefix.to_string(), obj.clone());
            }
        }
    }

    fields
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

        // Parse schema
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

        // Parse expected
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

        // Parse predicted
        let predicted_json: Value = match serde_json::from_str(&predicted) {
            Ok(value) => value,
            Err(err) => {
                return FeedbackMetric::new(
                    0.0,
                    format!("Generated output was not valid JSON ({err}). Input: {input_text}"),
                );
            }
        };

        let config = &self.metrics_config;
        let mut score: f32 = 0.0;
        let mut feedback_lines = Vec::new();

        // Step 1: Valid JSON
        score += config.base_parse_score;
        feedback_lines.push("󰸞 Valid JSON".to_string());

        // Step 2: Schema validation
        let compiled = match JSONSchema::compile(&schema_json) {
            Ok(schema) => schema,
            Err(err) => {
                return FeedbackMetric::new(
                    score * 0.5,
                    format!(
                        "Failed to compile schema: {err}. Ensure the schema is valid JSON Schema."
                    ),
                );
            }
        };

        if let Err(errors) = compiled.validate(&predicted_json) {
            feedback_lines.push(" Schema validation failed:".to_string());
            for error in errors.take(5) {
                feedback_lines.push(format!("  - {error}"));
            }
            return FeedbackMetric::new(score * 0.5, feedback_lines.join("\n"));
        }

        score += config.base_schema_score;
        feedback_lines.push("󰸞 Passes schema validation".to_string());

        // Step 3: Exact match check
        if predicted_json == expected_json {
            return FeedbackMetric::new(1.0, "󰸞 Exact match!".to_string());
        }

        // Step 4: Field-level precision/recall
        let expected_fields = get_all_fields(&expected_json, "");
        let predicted_fields = get_all_fields(&predicted_json, "");

        if expected_fields.is_empty() {
            return FeedbackMetric::new(score, feedback_lines.join("\n"));
        }

        let mut correct = 0;
        let mut wrong = 0;

        // Count correct and wrong fields
        for (field_path, expected_value) in &expected_fields {
            if let Some(predicted_value) = predicted_fields.get(field_path) {
                if predicted_value == expected_value {
                    correct += 1;
                } else {
                    wrong += 1;
                }
            }
            // Missing fields don't count as wrong
        }

        // Count extra/hallucinated fields
        let extra = predicted_fields
            .keys()
            .filter(|k| !expected_fields.contains_key(*k))
            .count();

        if predicted_fields.is_empty() {
            feedback_lines.push(" No fields extracted".to_string());
            return FeedbackMetric::new(score, feedback_lines.join("\n"));
        }

        // Calculate precision: correct / (correct + wrong + weighted_extra)
        let denom = correct + wrong + ((config.extra_field_weight * extra as f32) as usize);
        let precision = if denom > 0 {
            correct as f32 / denom as f32
        } else {
            0.0
        };

        // Calculate recall: correct / expected
        let recall = correct as f32 / expected_fields.len() as f32;

        // Use F-beta score to combine precision and recall
        let field_quality = f_beta(precision, recall, config.beta);
        let coverage_bonus = recall;

        // Final score: base + field quality + coverage bonus
        score += (config.field_weight * field_quality) + (config.coverage_weight * coverage_bonus);
        score = score.min(1.0);

        // Feedback
        feedback_lines.push(format!(
            "📊 Fields: {correct} correct, {wrong} wrong, {} missing, {extra} extra",
            expected_fields.len() - correct - wrong
        ));
        feedback_lines.push(format!(
            "📈 Precision: {precision:.2} | Recall: {recall:.2} | Score: {score:.3}"
        ));

        if wrong > 0 {
            feedback_lines.push(format!(
                "  {wrong} incorrect field(s) - verify extraction logic"
            ));
        }
        if extra > 0 {
            feedback_lines.push(format!(
                "  {extra} hallucinated field(s) - avoid adding unsupported fields"
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
    pub metrics_config: MetricsConfig,
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
            metrics_config,
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
        let logger =
            crate::mlflow_logger::GepaLogger::new(task.name.clone(), local_logging, log_dir)?;

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

        let mut models = models;

        // Disable caching so that prompt mutations are reflected immediately during the run.
        models.teacher.lm.cache = false;
        models.student.lm.cache = false;
        for fallback in &mut models.fallbacks {
            fallback.lm.cache = false;
        }

        models.student.configure_global();

        let mut program = ExtractionProgram::with_metrics_config(metrics_config);
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
            let mut handle = build_model_handle(&descriptor).await?;
            handle.lm.cache = false;
            Some(handle.lm)
        } else {
            Some(models.teacher.lm.clone())
        };

        let gepa = match (
            optimization.max_rollouts,
            optimization.max_lm_calls > 0,
            prompt_model,
        ) {
            (Some(max_rollouts), true, Some(pm)) => GEPA::builder()
                .num_iterations(usize::max(1, optimization.iterations as usize))
                .minibatch_size(usize::max(1, optimization.batch_size as usize))
                .num_trials(usize::max(1, optimization.rollouts_per_iteration as usize))
                .temperature(optimization.temperature)
                .track_stats(optimization.track_stats)
                .track_best_outputs(optimization.track_best_outputs)
                .max_rollouts(max_rollouts as usize)
                .max_lm_calls(optimization.max_lm_calls as usize)
                .prompt_model(pm)
                .build(),
            (Some(max_rollouts), true, None) => GEPA::builder()
                .num_iterations(usize::max(1, optimization.iterations as usize))
                .minibatch_size(usize::max(1, optimization.batch_size as usize))
                .num_trials(usize::max(1, optimization.rollouts_per_iteration as usize))
                .temperature(optimization.temperature)
                .track_stats(optimization.track_stats)
                .track_best_outputs(optimization.track_best_outputs)
                .max_rollouts(max_rollouts as usize)
                .max_lm_calls(optimization.max_lm_calls as usize)
                .build(),
            (Some(max_rollouts), false, Some(pm)) => GEPA::builder()
                .num_iterations(usize::max(1, optimization.iterations as usize))
                .minibatch_size(usize::max(1, optimization.batch_size as usize))
                .num_trials(usize::max(1, optimization.rollouts_per_iteration as usize))
                .temperature(optimization.temperature)
                .track_stats(optimization.track_stats)
                .track_best_outputs(optimization.track_best_outputs)
                .max_rollouts(max_rollouts as usize)
                .prompt_model(pm)
                .build(),
            (Some(max_rollouts), false, None) => GEPA::builder()
                .num_iterations(usize::max(1, optimization.iterations as usize))
                .minibatch_size(usize::max(1, optimization.batch_size as usize))
                .num_trials(usize::max(1, optimization.rollouts_per_iteration as usize))
                .temperature(optimization.temperature)
                .track_stats(optimization.track_stats)
                .track_best_outputs(optimization.track_best_outputs)
                .max_rollouts(max_rollouts as usize)
                .build(),
            (None, true, Some(pm)) => GEPA::builder()
                .num_iterations(usize::max(1, optimization.iterations as usize))
                .minibatch_size(usize::max(1, optimization.batch_size as usize))
                .num_trials(usize::max(1, optimization.rollouts_per_iteration as usize))
                .temperature(optimization.temperature)
                .track_stats(optimization.track_stats)
                .track_best_outputs(optimization.track_best_outputs)
                .max_lm_calls(optimization.max_lm_calls as usize)
                .prompt_model(pm)
                .build(),
            (None, true, None) => GEPA::builder()
                .num_iterations(usize::max(1, optimization.iterations as usize))
                .minibatch_size(usize::max(1, optimization.batch_size as usize))
                .num_trials(usize::max(1, optimization.rollouts_per_iteration as usize))
                .temperature(optimization.temperature)
                .track_stats(optimization.track_stats)
                .track_best_outputs(optimization.track_best_outputs)
                .max_lm_calls(optimization.max_lm_calls as usize)
                .build(),
            (None, false, Some(pm)) => GEPA::builder()
                .num_iterations(usize::max(1, optimization.iterations as usize))
                .minibatch_size(usize::max(1, optimization.batch_size as usize))
                .num_trials(usize::max(1, optimization.rollouts_per_iteration as usize))
                .temperature(optimization.temperature)
                .track_stats(optimization.track_stats)
                .track_best_outputs(optimization.track_best_outputs)
                .prompt_model(pm)
                .build(),
            (None, false, None) => GEPA::builder()
                .num_iterations(usize::max(1, optimization.iterations as usize))
                .minibatch_size(usize::max(1, optimization.batch_size as usize))
                .num_trials(usize::max(1, optimization.rollouts_per_iteration as usize))
                .temperature(optimization.temperature)
                .track_stats(optimization.track_stats)
                .track_best_outputs(optimization.track_best_outputs)
                .build(),
        };
        let config = serde_json::json!({
            "task_name": task.name,
            "num_iterations": optimization.iterations,
            "minibatch_size": optimization.batch_size,
            "rollouts_per_iteration": optimization.rollouts_per_iteration,
            "max_lm_calls": optimization.max_lm_calls,
            "max_rollouts": optimization.max_rollouts,
            "temperature": optimization.temperature,
            "track_stats": optimization.track_stats,
            "track_best_outputs": optimization.track_best_outputs,
            "student_model": models.student.descriptor.name,
            "teacher_model": models.teacher.descriptor.name,
        });
        logger.log_config(&config)?;

        // Prepare evaluation minibatch (same as Python's eval_minibatch_size)
        let eval_batch_size = optimization.batch_size.min(train_examples.len() as u32) as usize;
        let eval_minibatch = &train_examples[..eval_batch_size];

        eprintln!("\n[DEBUG] Baseline evaluation on {eval_batch_size} examples...");

        // Get baseline instruction for debugging
        let baseline_instruction = OptimizableTrait::get_signature(&program.solver).instruction();
        eprintln!("[DEBUG] Baseline instruction: {baseline_instruction}");

        // Evaluate baseline performance
        let mut baseline_total = 0.0;
        for (i, example) in eval_minibatch.iter().enumerate() {
            let prediction = program.forward(example.clone()).await?;
            let score = program.feedback_metric(example, &prediction).await;
            baseline_total += score.score;
            eprintln!(
                "[DEBUG] Baseline example {}: score={:.3}",
                i + 1,
                score.score
            );
            if i < 2 {
                eprintln!(
                    "[DEBUG]   Input: {}",
                    example.get("input_text", None).as_str().unwrap_or("")
                );
                eprintln!(
                    "[DEBUG]   Predicted: {}",
                    prediction.get("output_json", None).as_str().unwrap_or("")
                );
                eprintln!("[DEBUG]   Feedback: {}", score.feedback);
            }
        }
        let baseline_score = baseline_total / eval_batch_size as f32;
        eprintln!("[DEBUG] Baseline average score: {baseline_score:.3}\n");

        eprintln!("Running GEPA optimization...");
        let result = gepa
            .compile_with_feedback(&mut program, train_examples.clone())
            .await?;

        // Get optimized instruction for debugging
        let optimized_instruction = OptimizableTrait::get_signature(&program.solver).instruction();
        eprintln!("\n[DEBUG] Optimized instruction: {optimized_instruction}");

        eprintln!("[DEBUG] Final evaluation on {eval_batch_size} examples...");

        // Evaluate optimized performance on the same minibatch
        let mut optimized_total = 0.0;
        for (i, example) in eval_minibatch.iter().enumerate() {
            let prediction = program.forward(example.clone()).await?;
            let score = program.feedback_metric(example, &prediction).await;
            optimized_total += score.score;
            eprintln!(
                "[DEBUG] Optimized example {}: score={:.3}",
                i + 1,
                score.score
            );
            if i < 2 {
                eprintln!(
                    "[DEBUG]   Input: {}",
                    example.get("input_text", None).as_str().unwrap_or("")
                );
                eprintln!(
                    "[DEBUG]   Predicted: {}",
                    prediction.get("output_json", None).as_str().unwrap_or("")
                );
                eprintln!("[DEBUG]   Feedback: {}", score.feedback);
            }
        }
        let optimized_score = optimized_total / eval_batch_size as f32;
        let improvement = optimized_score - baseline_score;

        eprintln!("\n{}", "=".repeat(60));
        eprintln!("GEPA RESULTS");
        eprintln!("{}", "=".repeat(60));
        eprintln!(
            "Baseline score:  {:.3} ({:.1}%)",
            baseline_score,
            baseline_score * 100.0
        );
        eprintln!(
            "Optimized score: {:.3} ({:.1}%)",
            optimized_score,
            optimized_score * 100.0
        );
        eprintln!(
            "Improvement:     {:+.3} ({:+.1}%)",
            improvement,
            improvement * 100.0
        );
        eprintln!("{}\n", "=".repeat(60));

        persist_gepa_result(
            state_dir,
            &task.name,
            &result,
            baseline_score,
            optimized_score,
            train_examples.len(),
        )?;

        logger.log_final_result(&result)?;
        logger.finish()?;

        Ok(GepaOutcome {
            best_instruction: result.best_candidate.instruction.clone(),
            best_score: optimized_score,
            total_iterations: result.evolution_history.len(),
            total_rollouts: result.total_rollouts,
            total_lm_calls: result.total_lm_calls,
        })
    }
}

fn persist_gepa_result(
    state_dir: &Path,
    task_name: &str,
    result: &GEPAResult,
    baseline_score: f32,
    optimized_score: f32,
    num_examples: usize,
) -> Result<()> {
    use chrono::Local;

    // Create dated subdirectory structure: optimizations/YYYY-MM-DD/task_YYYYMMDD_HHMMSS/
    let now = Local::now();
    let date_str = now.format("%Y-%m-%d").to_string();
    let timestamp_str = now.format("%Y%m%d_%H%M%S").to_string();
    let run_name = format!("{task_name}_{timestamp_str}");

    let optimizations_dir = state_dir
        .join("optimizations")
        .join(&date_str)
        .join(&run_name);
    fs::create_dir_all(&optimizations_dir)?;

    // Save config.json - optimization parameters
    let config = serde_json::json!({
        "task": task_name,
        "timestamp": now.to_rfc3339(),
        "date": date_str,
        "run_name": run_name,
        "num_examples": num_examples,
    });
    fs::write(
        optimizations_dir.join("config.json"),
        serde_json::to_string_pretty(&config)?,
    )?;

    // Save metrics.json - scores and improvement
    let improvement = optimized_score - baseline_score;
    let metrics = serde_json::json!({
        "baseline_score": baseline_score,
        "optimized_score": optimized_score,
        "improvement": improvement,
        "improvement_percentage": improvement * 100.0,
        "total_rollouts": result.total_rollouts,
        "total_lm_calls": result.total_lm_calls,
        "num_examples": num_examples,
        "timestamp": now.to_rfc3339(),
    });
    fs::write(
        optimizations_dir.join("metrics.json"),
        serde_json::to_string_pretty(&metrics)?,
    )?;

    // Save optimized_instruction.txt - the optimized prompt
    fs::write(
        optimizations_dir.join("optimized_instruction.txt"),
        &result.best_candidate.instruction,
    )?;

    // Save result.json - full GEPA result with evolution history
    let serialized = serde_json::to_vec_pretty(result)?;
    fs::write(optimizations_dir.join("result.json"), serialized)?;

    // Save summary.txt - human-readable summary
    let summary = format!(
        "GEPA Optimization Summary\n\
         {}\n\n\
         Task: {}\n\
         Date: {}\n\
         Examples: {}\n\n\
         Results:\n\
         Baseline score:  {:.3} ({:.1}%)\n\
         Optimized score: {:.3} ({:.1}%)\n\
         Improvement:     {:+.3} ({:+.1}%)\n\n\
         Statistics:\n\
         Total rollouts: {}\n\
         Total LM calls: {}\n\n\
         {}\n\
         Optimized Instruction\n\
         {}\n\
         {}\n",
        "=".repeat(60),
        task_name,
        now.format("%Y-%m-%d %H:%M:%S"),
        num_examples,
        baseline_score,
        baseline_score * 100.0,
        optimized_score,
        optimized_score * 100.0,
        improvement,
        improvement * 100.0,
        result.total_rollouts,
        result.total_lm_calls,
        "=".repeat(60),
        "-".repeat(60),
        result.best_candidate.instruction,
    );
    fs::write(optimizations_dir.join("summary.txt"), summary)?;

    // Update the active prompt in prompts/task_name/
    let prompts_dir = state_dir.join("prompts").join(task_name);
    fs::create_dir_all(&prompts_dir)?;

    // Copy optimized instruction to active location
    fs::write(
        prompts_dir.join("optimized_instruction.txt"),
        &result.best_candidate.instruction,
    )?;

    // Save metadata.json to track which run is active
    let metadata = serde_json::json!({
        "active_run": format!("{}/{}", date_str, run_name),
        "timestamp": now.to_rfc3339(),
        "baseline_score": baseline_score,
        "optimized_score": optimized_score,
        "improvement": improvement,
    });
    fs::write(
        prompts_dir.join("metadata.json"),
        serde_json::to_string_pretty(&metadata)?,
    )?;

    eprintln!("\n✓ Saved optimization results to:");
    eprintln!("  {}", optimizations_dir.display());
    eprintln!("\n✓ Updated active prompt at:");
    eprintln!("  {}", prompts_dir.display());

    Ok(())
}

pub fn load_best_instruction(paths: &AppPaths, task_name: &str) -> Option<String> {
    // Try new structure first: prompts/task_name/optimized_instruction.txt
    let new_path = paths
        .state_dir
        .join("prompts")
        .join(task_name)
        .join("optimized_instruction.txt");

    if let Ok(content) = fs::read_to_string(&new_path) {
        return Some(content);
    }

    // Fallback to old structure: prompts/task_name/best.txt
    let old_path = paths
        .state_dir
        .join("prompts")
        .join(task_name)
        .join("best.txt");

    fs::read_to_string(old_path).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_get_all_fields_simple() {
        let obj = json!({
            "name": "John",
            "age": 30,
            "city": "NYC"
        });

        let fields = get_all_fields(&obj, "");

        assert_eq!(fields.len(), 3);
        assert_eq!(fields.get("name"), Some(&json!("John")));
        assert_eq!(fields.get("age"), Some(&json!(30)));
        assert_eq!(fields.get("city"), Some(&json!("NYC")));
    }

    #[test]
    fn test_get_all_fields_nested() {
        let obj = json!({
            "user": {
                "name": "John",
                "contact": {
                    "email": "john@example.com"
                }
            }
        });

        let fields = get_all_fields(&obj, "");

        assert_eq!(fields.len(), 2);
        assert_eq!(fields.get("user.name"), Some(&json!("John")));
        assert_eq!(
            fields.get("user.contact.email"),
            Some(&json!("john@example.com"))
        );
    }

    #[test]
    fn test_get_all_fields_array() {
        let obj = json!({
            "items": [
                {"id": 1, "name": "A"},
                {"id": 2, "name": "B"}
            ]
        });

        let fields = get_all_fields(&obj, "");

        assert_eq!(fields.len(), 4);
        assert_eq!(fields.get("items[0].id"), Some(&json!(1)));
        assert_eq!(fields.get("items[0].name"), Some(&json!("A")));
        assert_eq!(fields.get("items[1].id"), Some(&json!(2)));
        assert_eq!(fields.get("items[1].name"), Some(&json!("B")));
    }

    #[test]
    fn test_metric_scoring() {
        // This is a conceptual test - actual integration would require full Example/Prediction setup
        // Here we just verify the scoring logic

        // Perfect match scenario
        let expected = json!({"name": "John", "age": 30, "city": "NYC"});
        let predicted = json!({"name": "John", "age": 30, "city": "NYC"});

        let expected_fields = get_all_fields(&expected, "");
        let predicted_fields = get_all_fields(&predicted, "");

        let mut correct = 0;
        for (k, v) in &expected_fields {
            if predicted_fields.get(k) == Some(v) {
                correct += 1;
            }
        }

        assert_eq!(correct, 3);
        let precision = correct as f32 / predicted_fields.len() as f32;
        let recall = correct as f32 / expected_fields.len() as f32;
        assert_eq!(precision, 1.0);
        assert_eq!(recall, 1.0);

        // One wrong field scenario
        let predicted_wrong = json!({"name": "John", "age": 25, "city": "NYC"});
        let predicted_fields_wrong = get_all_fields(&predicted_wrong, "");

        let mut correct_wrong = 0;
        let mut wrong = 0;
        for (k, v) in &expected_fields {
            if let Some(pv) = predicted_fields_wrong.get(k) {
                if pv == v {
                    correct_wrong += 1;
                } else {
                    wrong += 1;
                }
            }
        }

        assert_eq!(correct_wrong, 2);
        assert_eq!(wrong, 1);
        let precision_wrong = correct_wrong as f32 / (correct_wrong + wrong) as f32;
        assert!((precision_wrong - 0.666).abs() < 0.01); // ~0.67

        // One missing field scenario (better than wrong!)
        let predicted_missing = json!({"name": "John", "city": "NYC"});
        let predicted_fields_missing = get_all_fields(&predicted_missing, "");

        let mut correct_missing = 0;
        for (k, v) in &expected_fields {
            if predicted_fields_missing.get(k) == Some(v) {
                correct_missing += 1;
            }
        }

        assert_eq!(correct_missing, 2);
        let precision_missing = correct_missing as f32 / predicted_fields_missing.len() as f32;
        assert_eq!(precision_missing, 1.0); // Perfect precision!
    }
}
