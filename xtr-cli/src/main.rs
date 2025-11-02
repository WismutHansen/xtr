use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use clap::Subcommand;
use clap::ValueEnum;
use std::io::Read;
use std::io::{self};
use xtr_core::ExtractionEngine;
use xtr_core::config::FeedbackModelOverrides;
use xtr_core::config::OptimizationSettings;

mod commands;
mod create;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliValidationMode {
    None,
    Warn,
    Error,
}

impl From<CliValidationMode> for xtr_core::ValidationMode {
    fn from(mode: CliValidationMode) -> Self {
        match mode {
            CliValidationMode::None => xtr_core::ValidationMode::None,
            CliValidationMode::Warn => xtr_core::ValidationMode::Warn,
            CliValidationMode::Error => xtr_core::ValidationMode::Error,
        }
    }
}

#[cfg(unix)]
fn increase_fd_limit() -> Result<()> {
    use anyhow::anyhow;

    let mut limits = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };

    unsafe {
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut limits) != 0 {
            return Err(anyhow!("failed to get file descriptor limit"));
        }

        limits.rlim_cur = if limits.rlim_max == libc::RLIM_INFINITY {
            10240
        } else {
            limits.rlim_max.min(10240)
        };

        if libc::setrlimit(libc::RLIMIT_NOFILE, &limits) != 0 {
            eprintln!("Warning: failed to increase file descriptor limit");
        }
    }

    Ok(())
}

#[cfg(not(unix))]
fn increase_fd_limit() -> Result<()> {
    Ok(())
}

#[derive(Parser)]
#[command(name = "xtr-cli")]
#[command(about = "Extraction and optimization CLI", long_about = None)]
#[command(arg_required_else_help = true)]
struct Cli {
    #[arg(long, short, global = true, help = "Show verbose debug output")]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[command(about = "Show configuration and task information")]
    Info,

    #[command(about = "Create resources (schemas, tasks, etc.)")]
    Create {
        #[command(subcommand)]
        resource: CreateCommands,
    },

    #[command(about = "Run GEPA optimization for a task")]
    Optimize {
        #[arg(help = "Task name to optimize (uses first configured task if omitted)")]
        task: Option<String>,

        #[arg(long, help = "Number of optimization iterations (default: 4)")]
        iterations: Option<u32>,

        #[arg(long, help = "Batch size for evaluation (default: 4)")]
        batch_size: Option<u32>,

        #[arg(long, help = "Number of trials per iteration (default: 6)")]
        rollouts_per_iteration: Option<u32>,

        #[arg(long, help = "Maximum total LM calls budget")]
        max_lm_calls: Option<u32>,

        #[arg(long, help = "Maximum total rollouts budget")]
        max_rollouts: Option<u32>,

        #[arg(
            long,
            help = "Temperature for LLM mutations (default: 0.9, range: 0.0-2.0)"
        )]
        temperature: Option<f32>,

        #[arg(long, help = "Track detailed optimization statistics (default: true)")]
        track_stats: Option<bool>,

        #[arg(
            long,
            help = "Track best outputs for inference-time search (default: false)"
        )]
        track_best_outputs: Option<bool>,
    },

    #[command(about = "Run inference on input text")]
    Get {
        #[arg(help = "Task name")]
        task: String,

        #[arg(long, help = "Input text to process (reads from stdin if omitted)")]
        text: Option<String>,

        #[arg(long, help = "Optional additional context")]
        context: Option<String>,

        #[arg(
            long,
            value_enum,
            default_value = "none",
            help = "Schema validation mode"
        )]
        validate: CliValidationMode,

        #[arg(
            long,
            short = 'r',
            default_value = "0",
            help = "Number of retry attempts on validation failure"
        )]
        retry: u32,

        #[arg(
            long,
            short = 's',
            help = "Use shots mode (with chat history) instead of independent retries"
        )]
        shots: bool,

        #[arg(long, short = 'm', help = "Override max_tokens for this request")]
        max_tokens: Option<u32>,
    },

    #[command(about = "Set the active optimization run for a task")]
    Activate {
        #[arg(help = "Task name")]
        task: String,

        #[arg(help = "Run path (e.g., 2025-10-28/contact_details_20251028_141425)")]
        run: String,
    },

    #[command(about = "List optimization history for a task")]
    History {
        #[arg(help = "Task name (shows all tasks if omitted)")]
        task: Option<String>,

        #[arg(long, help = "Show detailed metrics for each run")]
        detailed: bool,
    },

    #[command(about = "Compare two optimization runs")]
    Compare {
        #[arg(help = "Task name")]
        task: String,

        #[arg(help = "First run path")]
        run1: String,

        #[arg(help = "Second run path")]
        run2: String,
    },

    #[command(about = "Clean old optimization runs")]
    Clean {
        #[arg(long, help = "Delete runs older than N days (default: 30)")]
        older_than: Option<u32>,

        #[arg(long, help = "Dry run - show what would be deleted")]
        dry_run: bool,
    },
}

#[derive(Subcommand)]
enum CreateCommands {
    #[command(about = "Generate a JSON schema from a JSON file")]
    Schema {
        #[arg(help = "Path to the JSON file")]
        input: String,

        #[arg(
            long,
            help = "Schema name (defaults to input filename without extension)"
        )]
        name: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    increase_fd_limit()?;

    let cli = Cli::parse();
    let engine = ExtractionEngine::load("xtr")?;

    match cli.command {
        Commands::Create { resource } => match resource {
            CreateCommands::Schema { input, name } => {
                create::handle_create_schema(&input, name.as_deref())?;
            }
        },
        Commands::Optimize {
            task,
            iterations,
            batch_size,
            rollouts_per_iteration,
            max_lm_calls,
            max_rollouts,
            temperature,
            track_stats,
            track_best_outputs,
        } => {
            let task_name = task
                .or_else(|| engine.config().tasks.keys().next().cloned())
                .context("no task provided and configuration has no tasks")?;

            let mut overrides = None;
            if iterations.is_some()
                || batch_size.is_some()
                || rollouts_per_iteration.is_some()
                || max_lm_calls.is_some()
                || max_rollouts.is_some()
                || temperature.is_some()
                || track_stats.is_some()
                || track_best_outputs.is_some()
            {
                overrides = Some(OptimizationSettings {
                    iterations,
                    rollouts_per_iteration,
                    max_lm_calls,
                    batch_size,
                    max_rollouts,
                    temperature,
                    track_stats,
                    track_best_outputs,
                    feedback_models: FeedbackModelOverrides::default(),
                });
            }

            println!("Running GEPA optimization for task '{task_name}'...");
            let outcome = engine
                .optimize_task_with_overrides(&task_name, overrides)
                .await?;
            println!(
                "Best score: {:.3} after {} generations ({} rollouts, {} LM calls)",
                outcome.best_score,
                outcome.total_iterations,
                outcome.total_rollouts,
                outcome.total_lm_calls
            );
            println!("Optimized instruction saved to state directory.");
        }
        Commands::Get {
            task,
            text,
            context,
            validate,
            retry,
            shots,
            max_tokens,
        } => {
            let input_text = match text {
                Some(t) => t,
                None => {
                    let mut buffer = String::new();
                    io::stdin()
                        .read_to_string(&mut buffer)
                        .context("failed to read from stdin")?;
                    buffer
                }
            };

            let validation_mode = validate.into();
            let retry_mode = if shots {
                xtr_core::RetryMode::WithHistory
            } else {
                xtr_core::RetryMode::Independent
            };

            let output = engine
                .run_inference_with_options(
                    &task,
                    &input_text,
                    context.as_deref(),
                    cli.verbose,
                    validation_mode,
                    retry,
                    retry_mode,
                    max_tokens,
                )
                .await?;

            if cli.verbose {
                println!("{output}");
            } else {
                match serde_json::from_str::<serde_json::Value>(&output) {
                    Ok(json) => println!("{}", serde_json::to_string_pretty(&json)?),
                    Err(_) => println!("{output}"),
                }
            }
        }
        Commands::Activate { task, run } => {
            commands::handle_activate(&engine, &task, &run)?;
        }
        Commands::History { task, detailed } => {
            commands::handle_history(&engine, task.as_deref(), detailed)?;
        }
        Commands::Compare { task, run1, run2 } => {
            commands::handle_compare(&engine, &task, &run1, &run2)?;
        }
        Commands::Clean {
            older_than,
            dry_run,
        } => {
            commands::handle_clean(&engine, older_than.unwrap_or(30), dry_run)?;
        }
        Commands::Info => {
            println!(
                "Loaded configuration from {}",
                engine.paths().config_file.display()
            );

            if let Some(first_task) = engine.config().tasks.keys().next() {
                println!("Configured task detected: {}", first_task);
                let task = engine.resolve_task(first_task)?;
                println!(
                    "Teacher model: {}, Student model: {} ({} fallbacks)",
                    task.models.teacher.name,
                    task.models.student.name,
                    task.models.fallbacks.len()
                );
            } else {
                println!("No tasks configured yet. Add one to run optimization.");
            }
        }
    }

    Ok(())
}
