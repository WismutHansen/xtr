use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use config::Config as ConfigLoader;
use config::Environment;
use config::File;
use serde::Deserialize;
use serde::Serialize;

/// Embedded template used to bootstrap the on-disk configuration when the user
/// runs the tool for the first time.
pub const DEFAULT_CONFIG_TEMPLATE: &str = include_str!("../../examples/config.toml");

/// Container returned after loading configuration data and resolving runtime
/// paths.
#[derive(Debug, Clone)]
pub struct ConfigBundle {
    pub config: AppConfig,
    pub paths: AppPaths,
}

/// Resolve and load the configuration for the provided application name. If no
/// config file exists yet, a default file is created from
/// [`DEFAULT_CONFIG_TEMPLATE`].
pub fn load_or_initialize_config(app_name: impl AsRef<str>) -> Result<ConfigBundle> {
    let app_name = app_name.as_ref();
    let mut paths = AppPaths::discover(app_name)?;
    paths.ensure_config_dir()?;

    if !paths.config_file.exists() {
        if let Some(parent) = paths.config_file.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create config directory {}", parent.display())
            })?;
        }

        fs::write(&paths.config_file, DEFAULT_CONFIG_TEMPLATE).with_context(|| {
            format!(
                "failed to write default config to {}",
                paths.config_file.display()
            )
        })?;
    }

    let env_prefix = app_name
        .chars()
        .map(|ch| if ch == '-' { '_' } else { ch })
        .collect::<String>()
        .to_ascii_uppercase();

    let builder = ConfigLoader::builder()
        .add_source(File::from(paths.config_file.clone()))
        .add_source(
            Environment::with_prefix(&env_prefix)
                .separator("__")
                .try_parsing(true),
        );

    let mut config: AppConfig = builder
        .build()
        .with_context(|| {
            format!(
                "failed to parse configuration at {}",
                paths.config_file.display()
            )
        })?
        .try_deserialize()
        .context("failed to deserialize configuration into AppConfig")?;

    paths = paths.apply_storage_overrides(&config.storage)?;
    paths.ensure_runtime_dirs()?;

    config.normalize()?;

    Ok(ConfigBundle { config, paths })
}

/// Persistent runtime paths derived from XDG environment variables or sensible
/// fallbacks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppPaths {
    pub app_name: String,
    pub config_dir: PathBuf,
    pub config_file: PathBuf,
    pub data_dir: PathBuf,
    pub state_dir: PathBuf,
    pub cache_dir: PathBuf,
}

impl AppPaths {
    pub fn discover(app_name: impl Into<String>) -> Result<Self> {
        let app_name = app_name.into();
        let home = home_dir().context("unable to determine home directory for XDG resolution")?;

        let config_base = xdg_dir("XDG_CONFIG_HOME", &home, ".config");
        let data_base = xdg_dir("XDG_DATA_HOME", &home, ".local/share");
        let state_base = xdg_dir("XDG_STATE_HOME", &home, ".local/state");
        let cache_base = env::var("XDG_CACHE_HOME")
            .ok()
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| state_base.join("cache"));

        let config_dir = config_base.join(&app_name);
        let data_dir = data_base.join(&app_name);
        let state_dir = state_base.join(&app_name);
        let cache_dir = cache_base.join(&app_name);
        let config_file = config_dir.join("config.toml");

        Ok(Self {
            app_name,
            config_dir,
            config_file,
            data_dir,
            state_dir,
            cache_dir,
        })
    }

    pub fn ensure_config_dir(&self) -> Result<()> {
        fs::create_dir_all(&self.config_dir).with_context(|| {
            format!(
                "failed to create config directory {}",
                self.config_dir.display()
            )
        })
    }

    pub fn ensure_runtime_dirs(&self) -> Result<()> {
        for dir in [&self.data_dir, &self.state_dir, &self.cache_dir] {
            fs::create_dir_all(dir)
                .with_context(|| format!("failed to create runtime directory {}", dir.display()))?;
        }
        Ok(())
    }

    pub fn apply_storage_overrides(&self, storage: &StorageSettings) -> Result<Self> {
        let mut next = self.clone();

        if let Some(data_dir) = storage.data_dir.as_ref() {
            next.data_dir = resolve_path_value(data_dir, &self.config_dir)?;
        }

        if let Some(state_dir) = storage.state_dir.as_ref() {
            next.state_dir = resolve_path_value(state_dir, &self.config_dir)?;
        }

        if let Some(cache_dir) = storage.cache_dir.as_ref() {
            next.cache_dir = resolve_path_value(cache_dir, &self.config_dir)?;
        } else {
            // Ensure cache lives under the state directory by default.
            next.cache_dir = next.state_dir.join("cache");
        }

        Ok(next)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub models: ModelSection,
    pub tasks: HashMap<String, TaskConfig>,
    pub storage: StorageSettings,
    pub optimization: OptimizationSection,
    pub mlflow: MlflowSettings,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            models: ModelSection::default(),
            tasks: HashMap::new(),
            storage: StorageSettings::default(),
            optimization: OptimizationSection::default(),
            mlflow: MlflowSettings::default(),
        }
    }
}

impl AppConfig {
    pub fn normalize(&mut self) -> Result<()> {
        // Ensure defaults always have named teacher/student models.
        if self.models.defaults.teacher.name.is_none()
            || self.models.defaults.student.name.is_none()
        {
            bail!("default teacher and student models must specify a name");
        }
        Ok(())
    }

    pub fn resolve_task(&self, task_name: &str, paths: &AppPaths) -> Result<ResolvedTaskConfig> {
        let task_config = self.tasks.get(task_name);
        let model_overrides = self.models.tasks.get(task_name);
        let optimization_overrides = self.optimization.tasks.get(task_name);

        let models = self
            .models
            .resolve(task_name, model_overrides)
            .context("failed to resolve model configuration")?;

        let optimization = self
            .optimization
            .resolve(optimization_overrides, task_name)?;

        let schema_path = task_config
            .and_then(|task| task.schema.as_ref())
            .map(|path| resolve_path_value(path, &paths.config_dir))
            .transpose()
            .with_context(|| format!("failed to resolve schema path for task {}", task_name))?;

        let examples_dir = task_config
            .and_then(|task| task.examples.as_ref())
            .map(|path| resolve_path_value(path, &paths.config_dir))
            .transpose()
            .with_context(|| format!("failed to resolve examples path for task {}", task_name))?;

        Ok(ResolvedTaskConfig {
            name: task_name.to_string(),
            description: task_config.and_then(|task| task.description.clone()),
            schema_path,
            examples_dir,
            models,
            optimization,
            include_timestamp: task_config
                .and_then(|task| task.include_timestamp)
                .unwrap_or(false),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageSettings {
    pub data_dir: Option<String>,
    pub state_dir: Option<String>,
    pub cache_dir: Option<String>,
}

impl Default for StorageSettings {
    fn default() -> Self {
        Self {
            data_dir: None,
            state_dir: None,
            cache_dir: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MlflowSettings {
    pub tracking_uri: Option<String>,
    pub experiment_name: Option<String>,
    pub local_logging: bool,
    pub log_dir: Option<String>,
}

impl Default for MlflowSettings {
    fn default() -> Self {
        Self {
            tracking_uri: None,
            experiment_name: Some("xtr-optimization".to_string()),
            local_logging: true,
            log_dir: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TaskConfig {
    pub schema: Option<String>,
    pub examples: Option<String>,
    pub description: Option<String>,
    pub include_timestamp: Option<bool>,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            schema: None,
            examples: None,
            description: None,
            include_timestamp: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelSection {
    pub defaults: ModelDefaults,
    pub tasks: HashMap<String, TaskModelOverrides>,
}

impl Default for ModelSection {
    fn default() -> Self {
        Self {
            defaults: ModelDefaults::default(),
            tasks: HashMap::new(),
        }
    }
}

impl ModelSection {
    pub fn resolve(
        &self,
        task_name: &str,
        overrides: Option<&TaskModelOverrides>,
    ) -> Result<ResolvedModelConfig> {
        let merged_teacher = merge_model_specs(
            &self.defaults.teacher,
            overrides.and_then(|o| o.teacher.as_ref()),
        );
        let merged_student = merge_model_specs(
            &self.defaults.student,
            overrides.and_then(|o| o.student.as_ref()),
        );

        let teacher = merged_teacher.into_descriptor("teacher", task_name)?;
        let student = merged_student.into_descriptor("student", task_name)?;

        let mut fallback_specs = Vec::new();

        if let Some(override_list) = overrides {
            fallback_specs.extend(override_list.fallbacks.clone());
        }
        fallback_specs.extend(self.defaults.fallbacks.clone());

        let mut fallbacks = Vec::with_capacity(fallback_specs.len());
        for (index, spec) in fallback_specs.into_iter().enumerate() {
            let descriptor = spec
                .into_descriptor("fallback", task_name)
                .with_context(|| {
                    format!(
                        "failed to resolve fallback model {} for task {}",
                        index, task_name
                    )
                })?;
            fallbacks.push(descriptor);
        }

        Ok(ResolvedModelConfig {
            teacher,
            student,
            fallbacks,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelDefaults {
    pub teacher: ModelSpec,
    pub student: ModelSpec,
    pub fallbacks: Vec<ModelSpec>,
}

impl Default for ModelDefaults {
    fn default() -> Self {
        Self {
            teacher: ModelSpec {
                name: Some("llama-teacher".to_string()),
                base_url: Some("http://localhost:8080/v1".to_string()),
                api_key: None,
                adapter: Some("chat".to_string()),
                request_timeout_secs: Some(60),
            },
            student: ModelSpec {
                name: Some("llama-student".to_string()),
                base_url: Some("http://localhost:8081/v1".to_string()),
                api_key: None,
                adapter: Some("chat".to_string()),
                request_timeout_secs: Some(60),
            },
            fallbacks: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TaskModelOverrides {
    pub teacher: Option<ModelSpec>,
    pub student: Option<ModelSpec>,
    pub fallbacks: Vec<ModelSpec>,
}

impl Default for TaskModelOverrides {
    fn default() -> Self {
        Self {
            teacher: None,
            student: None,
            fallbacks: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelSpec {
    pub name: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub adapter: Option<String>,
    pub request_timeout_secs: Option<u64>,
}

impl Default for ModelSpec {
    fn default() -> Self {
        Self {
            name: None,
            base_url: None,
            api_key: None,
            adapter: None,
            request_timeout_secs: None,
        }
    }
}

impl ModelSpec {
    pub fn into_descriptor(self, role: &str, task_name: &str) -> Result<ModelDescriptor> {
        let name = self.name.ok_or_else(|| {
            anyhow!(
                "model '{}' for task '{}' is missing a name",
                role,
                task_name
            )
        })?;

        Ok(ModelDescriptor {
            name,
            base_url: self.base_url,
            api_key: self.api_key,
            adapter: self.adapter,
            request_timeout_secs: self.request_timeout_secs,
        })
    }
}

fn merge_model_specs(base: &ModelSpec, overrides: Option<&ModelSpec>) -> ModelSpec {
    let mut merged = base.clone();

    if let Some(override_spec) = overrides {
        if override_spec.name.is_some() {
            merged.name = override_spec.name.clone();
        }
        if override_spec.base_url.is_some() {
            merged.base_url = override_spec.base_url.clone();
        }
        if override_spec.api_key.is_some() {
            merged.api_key = override_spec.api_key.clone();
        }
        if override_spec.adapter.is_some() {
            merged.adapter = override_spec.adapter.clone();
        }
        if override_spec.request_timeout_secs.is_some() {
            merged.request_timeout_secs = override_spec.request_timeout_secs;
        }
    }

    merged
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OptimizationSection {
    pub defaults: OptimizationSettings,
    pub tasks: HashMap<String, OptimizationSettings>,
}

impl Default for OptimizationSection {
    fn default() -> Self {
        Self {
            defaults: OptimizationSettings::default(),
            tasks: HashMap::new(),
        }
    }
}

impl OptimizationSection {
    pub fn resolve(
        &self,
        overrides: Option<&OptimizationSettings>,
        task_name: &str,
    ) -> Result<ResolvedOptimizationSettings> {
        let merged = merge_optimization_settings(&self.defaults, overrides);

        let iterations = merged.iterations.unwrap_or(DEFAULT_ITERATIONS);
        let rollouts = merged
            .rollouts_per_iteration
            .unwrap_or(DEFAULT_ROLLOUTS_PER_ITERATION);
        let max_lm_calls = merged.max_lm_calls.unwrap_or(DEFAULT_MAX_LM_CALLS);
        let batch_size = merged.batch_size.unwrap_or(DEFAULT_BATCH_SIZE);
        let max_rollouts = merged.max_rollouts;
        let temperature = merged.temperature.unwrap_or(DEFAULT_TEMPERATURE);
        let track_stats = merged.track_stats.unwrap_or(DEFAULT_TRACK_STATS);
        let track_best_outputs = merged
            .track_best_outputs
            .unwrap_or(DEFAULT_TRACK_BEST_OUTPUTS);

        let feedback_teacher = merge_optional_models(
            self.defaults.feedback_models.teacher.as_ref(),
            overrides.and_then(|opt| opt.feedback_models.teacher.as_ref()),
        );
        let feedback_student = merge_optional_models(
            self.defaults.feedback_models.student.as_ref(),
            overrides.and_then(|opt| opt.feedback_models.student.as_ref()),
        );

        let teacher_feedback = feedback_teacher
            .map(|spec| spec.into_descriptor("feedback_teacher", task_name))
            .transpose()?;

        let student_feedback = feedback_student
            .map(|spec| spec.into_descriptor("feedback_student", task_name))
            .transpose()?;

        Ok(ResolvedOptimizationSettings {
            iterations,
            rollouts_per_iteration: rollouts,
            max_lm_calls,
            batch_size,
            max_rollouts,
            temperature,
            track_stats,
            track_best_outputs,
            feedback_models: ResolvedFeedbackModels {
                teacher: teacher_feedback,
                student: student_feedback,
            },
        })
    }
}

const DEFAULT_ITERATIONS: u32 = 4;
const DEFAULT_ROLLOUTS_PER_ITERATION: u32 = 6;
const DEFAULT_MAX_LM_CALLS: u32 = 32;
const DEFAULT_BATCH_SIZE: u32 = 4;
const DEFAULT_TEMPERATURE: f32 = 0.9;
const DEFAULT_TRACK_STATS: bool = true;
const DEFAULT_TRACK_BEST_OUTPUTS: bool = false;

pub fn merge_optimization_settings_public(
    base: &OptimizationSettings,
    overrides: Option<&OptimizationSettings>,
) -> OptimizationSettings {
    merge_optimization_settings(base, overrides)
}

fn merge_optimization_settings(
    base: &OptimizationSettings,
    overrides: Option<&OptimizationSettings>,
) -> OptimizationSettings {
    let mut merged = base.clone();
    if let Some(override_settings) = overrides {
        if override_settings.iterations.is_some() {
            merged.iterations = override_settings.iterations;
        }
        if override_settings.rollouts_per_iteration.is_some() {
            merged.rollouts_per_iteration = override_settings.rollouts_per_iteration;
        }
        if override_settings.max_lm_calls.is_some() {
            merged.max_lm_calls = override_settings.max_lm_calls;
        }
        if override_settings.batch_size.is_some() {
            merged.batch_size = override_settings.batch_size;
        }
        if override_settings.max_rollouts.is_some() {
            merged.max_rollouts = override_settings.max_rollouts;
        }
        if override_settings.temperature.is_some() {
            merged.temperature = override_settings.temperature;
        }
        if override_settings.track_stats.is_some() {
            merged.track_stats = override_settings.track_stats;
        }
        if override_settings.track_best_outputs.is_some() {
            merged.track_best_outputs = override_settings.track_best_outputs;
        }
        if override_settings.feedback_models.teacher.is_some() {
            merged.feedback_models.teacher = override_settings.feedback_models.teacher.clone();
        }
        if override_settings.feedback_models.student.is_some() {
            merged.feedback_models.student = override_settings.feedback_models.student.clone();
        }
    }
    merged
}

fn merge_optional_models(
    base: Option<&ModelSpec>,
    overrides: Option<&ModelSpec>,
) -> Option<ModelSpec> {
    match (base, overrides) {
        (None, None) => None,
        (Some(base_spec), None) => Some(base_spec.clone()),
        (None, Some(override_spec)) => Some(override_spec.clone()),
        (Some(base_spec), Some(override_spec)) => {
            Some(merge_model_specs(base_spec, Some(override_spec)))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OptimizationSettings {
    pub iterations: Option<u32>,
    pub rollouts_per_iteration: Option<u32>,
    pub max_lm_calls: Option<u32>,
    pub batch_size: Option<u32>,
    pub max_rollouts: Option<u32>,
    pub temperature: Option<f32>,
    pub track_stats: Option<bool>,
    pub track_best_outputs: Option<bool>,
    pub feedback_models: FeedbackModelOverrides,
}

impl Default for OptimizationSettings {
    fn default() -> Self {
        Self {
            iterations: Some(DEFAULT_ITERATIONS),
            rollouts_per_iteration: Some(DEFAULT_ROLLOUTS_PER_ITERATION),
            max_lm_calls: Some(DEFAULT_MAX_LM_CALLS),
            batch_size: Some(DEFAULT_BATCH_SIZE),
            max_rollouts: None,
            temperature: Some(DEFAULT_TEMPERATURE),
            track_stats: Some(DEFAULT_TRACK_STATS),
            track_best_outputs: Some(DEFAULT_TRACK_BEST_OUTPUTS),
            feedback_models: FeedbackModelOverrides::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FeedbackModelOverrides {
    pub teacher: Option<ModelSpec>,
    pub student: Option<ModelSpec>,
}

impl Default for FeedbackModelOverrides {
    fn default() -> Self {
        Self {
            teacher: None,
            student: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDescriptor {
    pub name: String,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub adapter: Option<String>,
    pub request_timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModelConfig {
    pub teacher: ModelDescriptor,
    pub student: ModelDescriptor,
    pub fallbacks: Vec<ModelDescriptor>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedFeedbackModels {
    pub teacher: Option<ModelDescriptor>,
    pub student: Option<ModelDescriptor>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedOptimizationSettings {
    pub iterations: u32,
    pub rollouts_per_iteration: u32,
    pub max_lm_calls: u32,
    pub batch_size: u32,
    pub max_rollouts: Option<u32>,
    pub temperature: f32,
    pub track_stats: bool,
    pub track_best_outputs: bool,
    pub feedback_models: ResolvedFeedbackModels,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedTaskConfig {
    pub name: String,
    pub description: Option<String>,
    pub schema_path: Option<PathBuf>,
    pub examples_dir: Option<PathBuf>,
    pub models: ResolvedModelConfig,
    pub optimization: ResolvedOptimizationSettings,
    pub include_timestamp: bool,
}

fn xdg_dir(var: &str, home: &Path, fallback_suffix: &str) -> PathBuf {
    env::var(var)
        .ok()
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(fallback_suffix))
}

fn resolve_path_value(value: &str, base_dir: &Path) -> Result<PathBuf> {
    let expanded = expand_path(value)?;
    let mut path = PathBuf::from(&expanded);
    if path.is_absolute() {
        path = path.components().collect();
        Ok(path)
    } else {
        Ok(base_dir.join(path))
    }
}

fn expand_path(value: &str) -> Result<String> {
    let home = home_dir();
    let home_utf8 = match home.as_ref() {
        Some(path) => Some(
            path.to_str()
                .ok_or_else(|| anyhow!("home directory contains invalid UTF-8"))?
                .to_string(),
        ),
        None => None,
    };

    let expanded = shellexpand::full_with_context(
        value,
        || home_utf8.as_deref(),
        |var| Ok(env::var(var).ok()),
    )
    .map_err(|error: shellexpand::LookupError<std::env::VarError>| {
        anyhow!("failed to expand '{}': {}", value, error)
    })?;
    Ok(expanded.into_owned())
}

fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("USERPROFILE").map(PathBuf::from))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::Path;
    use std::sync::Mutex;
    use std::sync::OnceLock;
    use tempfile::TempDir;

    fn set_env_path(var: &str, value: &Path) {
        // `std::env::set_var` is marked unsafe in Rust 1.88 because it mutates
        // global process state. Tests run in isolation, so we gate the call in a
        // single helper.
        unsafe { env::set_var(var, value) };
    }

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn creates_config_when_missing() {
        let _guard = env_lock().lock().unwrap();
        let tmp = TempDir::new().unwrap();
        let config_home = tmp.path().join("config");
        let data_home = tmp.path().join("data");
        let state_home = tmp.path().join("state");

        set_env_path("XDG_CONFIG_HOME", &config_home);
        set_env_path("XDG_DATA_HOME", &data_home);
        set_env_path("XDG_STATE_HOME", &state_home);

        let bundle = load_or_initialize_config("xtr-test").unwrap();

        assert!(
            bundle.paths.config_file.exists(),
            "config file not created at {}",
            bundle.paths.config_file.display()
        );
        assert!(bundle.paths.data_dir.exists());
        assert!(bundle.paths.state_dir.exists());
        assert!(bundle.paths.cache_dir.exists());

        let resolved = bundle
            .config
            .resolve_task("invoice_extraction", &bundle.paths)
            .unwrap();
        assert_eq!(resolved.name, "invoice_extraction");
    }

    #[test]
    fn respects_storage_overrides() {
        let _guard = env_lock().lock().unwrap();
        let tmp = TempDir::new().unwrap();
        let config_home = tmp.path().join("config");
        let data_home = tmp.path().join("data");
        let state_home = tmp.path().join("state");

        set_env_path("XDG_CONFIG_HOME", &config_home);
        set_env_path("XDG_DATA_HOME", &data_home);
        set_env_path("XDG_STATE_HOME", &state_home);

        let app_dir = config_home.join("xtr-override");
        fs::create_dir_all(&app_dir).unwrap();
        let config_file = app_dir.join("config.toml");
        let mut file = fs::File::create(&config_file).unwrap();
        writeln!(
            file,
            r#"
                [storage]
                data_dir = "~/custom/data"
                state_dir = "~/custom/state"
                cache_dir = "~/custom/state/cache"
            "#
        )
        .unwrap();

        let bundle = load_or_initialize_config("xtr-override").unwrap();

        let expanded_home = home_dir().unwrap();
        assert_eq!(bundle.paths.data_dir, expanded_home.join("custom/data"));
        assert_eq!(bundle.paths.state_dir, expanded_home.join("custom/state"));
        assert_eq!(
            bundle.paths.cache_dir,
            expanded_home.join("custom/state/cache")
        );
    }
}
