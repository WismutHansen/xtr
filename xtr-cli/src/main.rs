use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use clap::Subcommand;
use std::io::{self, Read};
use xtr_core::ExtractionEngine;
use xtr_core::config::FeedbackModelOverrides;
use xtr_core::config::OptimizationSettings;

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
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    increase_fd_limit()?;

    let cli = Cli::parse();
    let engine = ExtractionEngine::load("xtr")?;

    match cli.command {
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

            let output = engine
                .run_inference(&task, &input_text, context.as_deref(), cli.verbose)
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
