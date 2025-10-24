//! Core library for the XTR structured data extraction engine.

pub mod config;
pub mod engine;
pub mod examples;
pub mod lm;
pub mod optimization;
pub mod mlflow_logger;

pub use config::AppConfig;
pub use config::AppPaths;
pub use config::ConfigBundle;
pub use config::ModelDescriptor;
pub use config::OptimizationSettings;
pub use config::ResolvedModelConfig;
pub use config::ResolvedTaskConfig;
pub use config::StorageSettings;
pub use config::TaskConfig;
pub use config::load_or_initialize_config;
pub use engine::ExtractionEngine;
pub use examples::TaskExample;
pub use examples::load_task_examples;
pub use lm::AdapterKind;
pub use lm::ModelHandle;
pub use lm::TaskModelHandles;
pub use lm::build_model_handle;
pub use optimization::GepaOutcome;
pub use optimization::GepaRunner;
pub use optimization::load_best_instruction;
