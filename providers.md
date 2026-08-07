---
title: Supported Providers
description: Complete list of LLM providers supported by any-llm including OpenAI, Anthropic, Mistral, and more
---

`any-llm` supports multiple providers. Provider source code is in [`src/any_llm/providers/`](https://github.com/mozilla-ai/any-llm/tree/main/src/any_llm/providers).

Is your endpoint OpenAI-compatible but not listed below? You are not blocked: use [`AnyLLM.create_openai_compatible`](quickstart.md#custom-openai-compatible-endpoints). Prefer it over pointing the `openai` provider at a custom `api_base`, which misreports the provider identity as `openai`, silently sends any `OPENAI_API_KEY` in your environment to the custom endpoint, and rejects keyless local servers.

## Support tiers

The **Tier** column is a support promise, not a statement about how the provider is implemented:

- ✅ **Verified**: we hold an API key, integration tests run against the provider in CI, and we fix breakage.
- 🤝 **Community**: verified live by the contributor when it was added, then community-maintained. No CI key, so its integration tests skip in CI. Report breakage by opening an issue.

A provider can be verified whether it ships as a code folder or as a single config row, and a config-only gateway we hold a key for is verified.

| ID | Tier | Key | Base | Responses | Completion | Streaming<br>(Completions) | Reasoning<br>(Completions) | Image <br>(Completions) | Embedding | List Models | Batch |
|----|------|-----|------|-----------|------------|--------------------------|--------------------------|-----------|-----------|-------------|-------|
| [`anthropic`](https://docs.anthropic.com/en/home) | ✅ Verified | ANTHROPIC_API_KEY | ANTHROPIC_BASE_URL | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| [`atlascloud`](https://www.atlascloud.ai/docs) | 🤝 Community | ATLASCLOUD_API_KEY | ATLASCLOUD_API_BASE | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| [`azure`](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure) | 🤝 Community | AZURE_API_KEY | AZURE_AI_CHAT_ENDPOINT | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| [`azureanthropic`](https://learn.microsoft.com/en-us/azure/ai-foundry/model-inference/concepts/models) | 🤝 Community | AZURE_ANTHROPIC_API_KEY | AZURE_ANTHROPIC_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| [`azureopenai`](https://learn.microsoft.com/en-us/azure/ai-foundry/) | ✅ Verified | AZURE_OPENAI_API_KEY | AZURE_OPENAI_ENDPOINT | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| [`bedrock`](https://aws.amazon.com/bedrock/) | ✅ Verified | AWS_BEARER_TOKEN_BEDROCK | AWS_ENDPOINT_URL_BEDROCK_RUNTIME | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [`cascadia`](https://cascadia.to) | 🤝 Community | CASCADIA_API_KEY | CASCADIA_API_BASE | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| [`cerebras`](https://docs.cerebras.ai/) | ✅ Verified | CEREBRAS_API_KEY | CEREBRAS_API_BASE | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| [`cohere`](https://cohere.com/api) | ✅ Verified | COHERE_API_KEY | COHERE_BASE_URL | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`dashscope`](https://bailian.console.aliyun.com/cn-beijing/?tab=api#/api) | 🤝 Community | DASHSCOPE_API_KEY | DASHSCOPE_API_BASE | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| [`databricks`](https://docs.databricks.com/) | 🤝 Community | DATABRICKS_TOKEN | DATABRICKS_HOST | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| [`deepinfra`](https://deepinfra.com/docs/openai_api) | 🤝 Community | DEEPINFRA_API_KEY | DEEPINFRA_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`deepseek`](https://platform.deepseek.com/) | ✅ Verified | DEEPSEEK_API_KEY | DEEPSEEK_API_BASE | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| [`edenai`](https://www.edenai.co/docs) | 🤝 Community | EDENAI_API_KEY | EDENAI_API_BASE | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| [`fireworks`](https://fireworks.ai/api) | ✅ Verified | FIREWORKS_API_KEY | FIREWORKS_API_BASE | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| [`gemini`](https://ai.google.dev/gemini-api/docs) | ✅ Verified | GEMINI_API_KEY/GOOGLE_API_KEY | GOOGLE_GEMINI_BASE_URL | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [`github`](https://docs.github.com/en/github-models) | 🤝 Community | GITHUB_TOKEN | GITHUB_MODELS_API_BASE | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ |
| [`gmi`](https://docs.gmicloud.ai/inference-engine/api-reference/llm-api-reference) | 🤝 Community | GMI_API_KEY | GMI_API_BASE | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| [`groq`](https://groq.com/api) | ✅ Verified | GROQ_API_KEY | GROQ_BASE_URL | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| [`huggingface`](https://huggingface.co/docs/huggingface_hub/package_reference/inference_client) | 🤝 Community | HF_TOKEN | HUGGINGFACE_API_BASE | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| [`inception`](https://inceptionlabs.ai/) | ✅ Verified | INCEPTION_API_KEY | INCEPTION_API_BASE | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| [`kenari`](https://kenari.id/docs) | 🤝 Community | KENARI_API_KEY | KENARI_API_BASE | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| [`llama`](https://www.llama.com/products/llama-api/) | 🤝 Community | LLAMA_API_KEY | LLAMA_API_BASE | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| [`llamacpp`](https://github.com/ggml-org/llama.cpp) | ✅ Verified | None | LLAMACPP_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`llamafile`](https://github.com/Mozilla-Ocho/llamafile) | ✅ Verified | None | LLAMAFILE_API_BASE | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ |
| [`lmstudio`](https://lmstudio.ai/docs/python) | ✅ Verified | LM_STUDIO_API_KEY | LM_STUDIO_API_BASE | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| [`meta`](https://dev.meta.ai/docs) | 🤝 Community | MODEL_API_KEY | META_API_BASE | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ |
| [`minimax`](https://www.minimax.io/platform_overview) | ✅ Verified | MINIMAX_API_KEY | MINIMAX_API_BASE | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| [`mistral`](https://docs.mistral.ai/) | ✅ Verified | MISTRAL_API_KEY | MISTRAL_API_BASE | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| [`moonshot`](https://platform.moonshot.ai/) | ✅ Verified | MOONSHOT_API_KEY | MOONSHOT_API_BASE | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| [`mzai`](https://any-llm.ai) | 🤝 Community | ANY_LLM_KEY | ANY_LLM_PLATFORM_URL | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`nebius`](https://studio.nebius.ai/) | ✅ Verified | NEBIUS_API_KEY | NEBIUS_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`neosantara`](https://docs.neosantara.xyz) | 🤝 Community | NEOSANTARA_API_KEY | NEOSANTARA_API_BASE | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [`ollama`](https://github.com/ollama/ollama) | ✅ Verified | None | OLLAMA_HOST | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`openai`](https://platform.openai.com/docs/api-reference) | ✅ Verified | OPENAI_API_KEY | OPENAI_BASE_URL | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| [`openrouter`](https://openrouter.ai/docs) | ✅ Verified | OPENROUTER_API_KEY | OPENROUTER_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`otari`](https://mozilla-ai.github.io/otari/) | ✅ Verified | OTARI_API_KEY | OTARI_API_BASE | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [`perplexity`](https://docs.perplexity.ai/) | 🤝 Community | PERPLEXITY_API_KEY | PERPLEXITY_BASE_URL | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| [`portkey`](https://portkey.ai/docs) | ✅ Verified | PORTKEY_API_KEY | PORTKEY_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| [`qiniu`](https://developer.qiniu.com/aitokenapi) | 🤝 Community | QINIU_API_KEY | QINIU_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`requesty`](https://docs.requesty.ai) | 🤝 Community | REQUESTY_API_KEY | REQUESTY_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`sagemaker`](https://aws.amazon.com/sagemaker/) | 🤝 Community | AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY | SAGEMAKER_ENDPOINT_URL | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| [`sambanova`](https://sambanova.ai/) | ✅ Verified | SAMBANOVA_API_KEY | SAMBANOVA_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`telnyx`](https://developers.telnyx.com/docs/inference/getting-started) | 🤝 Community | TELNYX_API_KEY | TELNYX_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| [`together`](https://together.ai/) | ✅ Verified | TOGETHER_API_KEY | TOGETHER_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| [`vertexai`](https://cloud.google.com/vertex-ai/docs) | 🤝 Community |  | VERTEXAI_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [`vertexaianthropic`](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude) | 🤝 Community |  | VERTEXAI_ANTHROPIC_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| [`vllm`](https://docs.vllm.ai/) | 🤝 Community | VLLM_API_KEY | VLLM_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`voyage`](https://docs.voyageai.com/) | ✅ Verified | VOYAGE_API_KEY | VOYAGE_API_BASE | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| [`watsonx`](https://www.ibm.com/watsonx) | 🤝 Community | WATSONX_API_KEY | WATSONX_URL | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ |
| [`xai`](https://x.ai/) | ✅ Verified | XAI_API_KEY | XAI_API_BASE | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| [`zai`](https://docs.z.ai/guides/develop/python/introduction) | ✅ Verified | ZAI_API_KEY | ZAI_BASE_URL | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
