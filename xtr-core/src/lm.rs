use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use dspy_rs::Cache;
use dspy_rs::CallResult;
use dspy_rs::Chat;
use dspy_rs::ChatAdapter;
use dspy_rs::Example;
use dspy_rs::LM;
use dspy_rs::Message;
use dspy_rs::MetaSignature;
use dspy_rs::Prediction;
use dspy_rs::adapter::Adapter;
use dspy_rs::serde_utils::get_iter_from_value;
use rig::tool::ToolDyn;
use serde_json::Value;

use crate::config::ModelDescriptor;
use crate::config::ResolvedModelConfig;

#[derive(Default, Clone)]
struct JsonAdapter;

impl JsonAdapter {
    fn format_system_message(signature: &dyn MetaSignature) -> String {
        let base_instruction = signature.instruction();
        let mut message = if base_instruction.is_empty() {
            "You convert unstructured text into structured JSON that strictly matches the caller-provided schema. Only include fields supported by the schema and leave everything else out.".to_string()
        } else {
            base_instruction.clone()
        };

        message.push_str(
            "\n\nRespond with a single JSON object containing:\n  - \"reasoning\": brief natural-language justification (string)\n  - \"output_json\": the extracted data as a JSON string that validates against the provided schema.\nDo not add any surrounding commentary, markdown fences, or additional keys. The JSON must be the entire response.",
        );
        message
    }

    fn value_to_string(value: Value) -> String {
        match value {
            Value::String(s) => s,
            Value::Null => String::new(),
            other => other.to_string(),
        }
    }

    fn format_user_message(signature: &dyn MetaSignature, inputs: &Example) -> String {
        let mut sections = Vec::new();

        for (field_name, _) in get_iter_from_value(&signature.input_fields()) {
            let value = inputs.get(field_name.as_str(), None);
            let as_string = Self::value_to_string(value);

            if field_name.eq_ignore_ascii_case("schema") {
                sections.push(format!("Schema:\n```json\n{}\n```", as_string.trim()));
            } else {
                sections.push(format!("{field_name}:\n{as_string}"));
            }
        }

        sections.push(
            "Return only the JSON object described in the system message. It must be valid JSON without markdown fences."
                .to_string(),
        );

        sections.join("\n\n")
    }

    fn format_demo_messages(signature: &dyn MetaSignature, demo: &Example) -> (String, String) {
        let user = Self::format_user_message(signature, demo);

        let mut response_object = serde_json::Map::new();
        for (field_name, _) in get_iter_from_value(&signature.output_fields()) {
            let value = demo.get(field_name.as_str(), None);
            let as_string = Self::value_to_string(value);

            if field_name == "output_json" {
                response_object.insert(field_name.clone(), Value::String(as_string));
            } else {
                response_object.insert(field_name.clone(), Value::String(as_string));
            }
        }

        (user, Value::Object(response_object).to_string())
    }

    fn strip_think_tags(text: &str) -> String {
        let mut result = text.to_string();

        // Remove <think>...</think> tags (case insensitive, handles newlines)
        let re = regex::Regex::new(r"(?is)<think>.*?</think>").unwrap();
        result = re.replace_all(&result, "").to_string();

        result.trim().to_string()
    }

    fn extract_json_payload(raw: &str) -> Option<Value> {
        // First strip any <think> tags
        let cleaned = Self::strip_think_tags(raw);
        let trimmed = cleaned.trim();

        if trimmed.is_empty() {
            return None;
        }

        if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
            return Some(value);
        }

        if let Some(stripped) = Self::strip_code_fence(trimmed) {
            if let Ok(value) = serde_json::from_str::<Value>(&stripped) {
                return Some(value);
            }
        }

        if let Some(fragment) = Self::extract_braced_fragment(trimmed) {
            if let Ok(value) = serde_json::from_str::<Value>(&fragment) {
                return Some(value);
            }
        }

        None
    }

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

    fn extract_braced_fragment(text: &str) -> Option<String> {
        Self::extract_balanced_fragment(text, '{', '}')
    }

    fn extract_balanced_fragment(text: &str, open: char, close: char) -> Option<String> {
        let mut depth = 0usize;
        let mut start = None;

        for (idx, ch) in text.char_indices() {
            if ch == open {
                if start.is_none() {
                    start = Some(idx);
                }
                depth += 1;
            } else if ch == close {
                if depth == 0 {
                    continue;
                }
                depth -= 1;
                if depth == 0 {
                    let begin = start?;
                    return Some(text[begin..=idx].to_string());
                }
            }
        }

        None
    }

    fn parse_response_payload(content: &str) -> HashMap<String, Value> {
        let mut data = HashMap::new();
        if let Some(payload) = Self::extract_json_payload(content) {
            if let Some(map) = payload.as_object() {
                if let Some(reasoning) = map.get("reasoning") {
                    data.insert(
                        "reasoning".to_string(),
                        if reasoning.is_string() {
                            reasoning.clone()
                        } else {
                            Value::String(reasoning.to_string())
                        },
                    );
                }

                if let Some(output_json) = map.get("output_json") {
                    let normalized = if output_json.is_string() {
                        output_json.clone()
                    } else {
                        Value::String(output_json.to_string())
                    };
                    data.insert("output_json".to_string(), normalized);
                    return data;
                }
            }

            // If payload is already the structured object, treat it as the extraction.
            data.insert(
                "output_json".to_string(),
                Value::String(payload.to_string()),
            );
        } else {
            data.insert(
                "output_json".to_string(),
                Value::String(content.trim().to_string()),
            );
        }

        data
    }
}

#[async_trait::async_trait]
impl Adapter for JsonAdapter {
    fn format(&self, signature: &dyn MetaSignature, inputs: Example) -> Chat {
        let mut chat = Chat::new(vec![]);
        chat.push("system", &Self::format_system_message(signature));

        let demos = signature.demos();
        for demo in demos {
            let (user, assistant) = Self::format_demo_messages(signature, &demo);
            chat.push("user", &user);
            chat.push("assistant", &assistant);
        }

        let user_message = Self::format_user_message(signature, &inputs);
        chat.push("user", &user_message);

        chat
    }

    fn parse_response(
        &self,
        _signature: &dyn MetaSignature,
        response: Message,
    ) -> HashMap<String, Value> {
        let content = response.content();
        Self::parse_response_payload(&content)
    }

    async fn call(
        &self,
        lm: Arc<LM>,
        signature: &dyn MetaSignature,
        inputs: Example,
        _tools: Vec<Arc<dyn ToolDyn>>,
    ) -> Result<Prediction> {
        if lm.cache
            && let Some(cache) = lm.cache_handler.as_ref()
        {
            let cache_key = inputs.clone();
            if let Some(cached) = cache.lock().await.get(cache_key).await? {
                return Ok(cached);
            }
        }

        let messages = self.format(signature, inputs.clone());
        let response = lm.call(messages, vec![]).await?;
        let prompt_str = response.chat.to_json().to_string();

        let data = self.parse_response(signature, response.output.clone());
        let prediction = Prediction {
            data,
            lm_usage: response.usage,
        };

        if lm.cache
            && let Some(cache) = lm.cache_handler.as_ref()
        {
            let (tx, rx) = tokio::sync::mpsc::channel(1);
            let cache_clone = cache.clone();
            let inputs_clone = inputs.clone();
            let prediction_clone = prediction.clone();

            tokio::spawn(async move {
                let _ = cache_clone.lock().await.insert(inputs_clone, rx).await;
            });

            tx.send(CallResult {
                prompt: prompt_str,
                prediction: prediction_clone,
            })
            .await
            .map_err(|_| anyhow!("Failed to send to cache"))?;
        }

        Ok(prediction)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AdapterKind {
    Chat,
    Json,
}

impl AdapterKind {
    fn from_descriptor(descriptor: &ModelDescriptor) -> Result<Self> {
        match descriptor
            .adapter
            .as_deref()
            .unwrap_or("json")
            .to_ascii_lowercase()
            .as_str()
        {
            "chat-legacy" | "legacy_chat" | "legacy-chat" => Ok(Self::Chat),
            "chat" | "json" | "structured_json" | "structured-json" => Ok(Self::Json),
            other => Err(anyhow!(
                "unsupported adapter '{other}' for model '{}'",
                descriptor.name
            )),
        }
    }

    pub fn install(self, lm: LM) {
        match self {
            Self::Chat => dspy_rs::configure(lm, ChatAdapter),
            Self::Json => dspy_rs::configure(lm, JsonAdapter),
        }
    }
}

#[derive(Clone)]
pub struct ModelHandle {
    pub descriptor: ModelDescriptor,
    pub adapter: AdapterKind,
    pub lm: LM,
}

impl ModelHandle {
    pub fn configure_global(&self) {
        self.adapter.install(self.lm.clone());
    }
}

pub async fn build_model_handle(descriptor: &ModelDescriptor) -> Result<ModelHandle> {
    let adapter = AdapterKind::from_descriptor(descriptor)?;
    let api_key = descriptor
        .api_key
        .clone()
        .unwrap_or_else(|| "not_needed".to_string());

    let base_url = descriptor
        .base_url
        .clone()
        .unwrap_or_else(|| "https://api.openai.com/v1".to_string());

    let lm = if let Some(max_tokens) = descriptor.max_tokens {
        LM::builder()
            .api_key(api_key)
            .base_url(base_url)
            .model(descriptor.name.clone())
            .max_tokens(max_tokens)
            .build()
            .await?
    } else {
        LM::builder()
            .api_key(api_key)
            .base_url(base_url)
            .model(descriptor.name.clone())
            .build()
            .await?
    };

    Ok(ModelHandle {
        descriptor: descriptor.clone(),
        adapter,
        lm,
    })
}

#[derive(Clone)]
pub struct TaskModelHandles {
    pub teacher: ModelHandle,
    pub student: ModelHandle,
    pub fallbacks: Vec<ModelHandle>,
}

impl TaskModelHandles {
    pub async fn load(config: &ResolvedModelConfig) -> Result<Self> {
        let teacher = build_model_handle(&config.teacher)
            .await
            .context("failed to build teacher LM")?;
        let student = build_model_handle(&config.student)
            .await
            .context("failed to build student LM")?;

        let mut fallbacks = Vec::with_capacity(config.fallbacks.len());
        for descriptor in &config.fallbacks {
            fallbacks.push(
                build_model_handle(descriptor).await.with_context(|| {
                    format!("failed to build fallback LM '{}'", descriptor.name)
                })?,
            );
        }

        Ok(Self {
            teacher,
            student,
            fallbacks,
        })
    }
}
