# Issues

## Open

### [trx-ev42] Vision support: accept image inputs and send to VLMs via multimodal API (P1, feature)
Add vision/multimodal support to xtr so it can accept image inputs (file paths or base64) alongside text and send them to VLMs (GPT-4o, Claude, Gemini, local vision models) using the OpenAI-compatible multimodal message format. This enables use cases like: ctx desktop screenshots being analyzed by xtr with a schema for structured screen state extraction. The image_paths field already exists in TaskExample but images are not yet included in LLM API calls. Key work: (1) encode images as base64 data URIs, (2) build multimodal content arrays in chat messages, (3) support --image flag in CLI for ad-hoc image input, (4) pipe support for ctx integration (ctx capture | xtr get screen_context --image stdin).

