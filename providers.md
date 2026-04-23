---
title: Supported Providers
description: Complete list of LLM providers supported by any-llm including OpenAI, Anthropic, Mistral, and more
---

`any-llm` supports multiple providers. Provider source code is in [`src/any_llm/providers/`](https://github.com/mozilla-ai/any-llm/tree/main/src/any_llm/providers).

| ID | Key | Base | Responses | Completion | Streaming<br>(Completions) | Reasoning<br>(Completions) | Image <br>(Completions) | Embedding | List Models | Batch |
|----|-----|------|-----------|------------|--------------------------|--------------------------|-----------|-----------|-------------|-------|
| [`anthropic`](https://docs.anthropic.com/en/home) | ANTHROPIC_API_KEY | ANTHROPIC_BASE_URL | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| [`azure`](https://azure.microsoft.com/en-us/products/ai-services/openai-service) | AZURE_API_KEY | AZURE_AI_CHAT_ENDPOINT | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| [`azureanthropic`](https://learn.microsoft.com/en-us/azure/ai-foundry/model-inference/concepts/models) | AZURE_ANTHROPIC_API_KEY | AZURE_ANTHROPIC_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| [`azureopenai`](https://learn.microsoft.com/en-us/azure/ai-foundry/) | AZURE_OPENAI_API_KEY | AZURE_OPENAI_ENDPOINT | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| [`bedrock`](https://aws.amazon.com/bedrock/) | AWS_BEARER_TOKEN_BEDROCK | AWS_ENDPOINT_URL_BEDROCK_RUNTIME | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`cerebras`](https://docs.cerebras.ai/) | CEREBRAS_API_KEY | CEREBRAS_API_BASE | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| [`cohere`](https://cohere.com/api) | COHERE_API_KEY | COHERE_BASE_URL | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| [`dashscope`](https://bailian.console.aliyun.com/cn-beijing/?tab=api#/api) | DASHSCOPE_API_KEY | DASHSCOPE_API_BASE | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| [`databricks`](https://docs.databricks.com/) | DATABRICKS_TOKEN | DATABRICKS_HOST | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| [`deepinfra`](https://deepinfra.com/docs/openai_api) | DEEPINFRA_API_KEY | DEEPINFRA_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`deepseek`](https://platform.deepseek.com/) | DEEPSEEK_API_KEY | DEEPSEEK_API_BASE | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| [`fireworks`](https://fireworks.ai/api) | FIREWORKS_API_KEY | FIREWORKS_API_BASE | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| [`gateway`](https://github.com/mozilla-ai/any-llm) | GATEWAY_API_KEY | GATEWAY_API_BASE | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [`gemini`](https://ai.google.dev/gemini-api/docs) | GEMINI_API_KEY/GOOGLE_API_KEY | GOOGLE_GEMINI_BASE_URL | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`groq`](https://groq.com/api) | GROQ_API_KEY | GROQ_BASE_URL | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| [`huggingface`](https://huggingface.co/docs/huggingface_hub/package_reference/inference_client) | HF_TOKEN | HUGGINGFACE_API_BASE | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| [`inception`](https://inceptionlabs.ai/) | INCEPTION_API_KEY | INCEPTION_API_BASE | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| [`llama`](https://www.llama.com/products/llama-api/) | LLAMA_API_KEY | LLAMA_API_BASE | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| [`llamacpp`](https://github.com/ggml-org/llama.cpp) | None | LLAMACPP_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`llamafile`](https://github.com/Mozilla-Ocho/llamafile) | None | LLAMAFILE_API_BASE | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ |
| [`lmstudio`](https://lmstudio.ai/) | LM_STUDIO_API_KEY | LM_STUDIO_API_BASE | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`minimax`](https://www.minimax.io/platform_overview) | MINIMAX_API_KEY | MINIMAX_API_BASE | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| [`mistral`](https://docs.mistral.ai/) | MISTRAL_API_KEY | MISTRAL_API_BASE | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| [`moonshot`](https://platform.moonshot.ai/) | MOONSHOT_API_KEY | MOONSHOT_API_BASE | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| [`mzai`](https://any-llm.ai) | ANY_LLM_KEY | ANY_LLM_PLATFORM_URL | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`nebius`](https://studio.nebius.ai/) | NEBIUS_API_KEY | NEBIUS_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`ollama`](https://github.com/ollama/ollama) | None | OLLAMA_HOST | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`openai`](https://platform.openai.com/docs/api-reference) | OPENAI_API_KEY | OPENAI_BASE_URL | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| [`openrouter`](https://openrouter.ai/docs) | OPENROUTER_API_KEY | OPENROUTER_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`perplexity`](https://docs.perplexity.ai/) | PERPLEXITY_API_KEY | PERPLEXITY_BASE_URL | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| [`portkey`](https://portkey.ai/docs) | PORTKEY_API_KEY | PORTKEY_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| [`sagemaker`](https://aws.amazon.com/sagemaker/) | AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY | SAGEMAKER_ENDPOINT_URL | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| [`sambanova`](https://sambanova.ai/) | SAMBANOVA_API_KEY | SAMBANOVA_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`together`](https://together.ai/) | TOGETHER_API_KEY | TOGETHER_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| [`vertexai`](https://cloud.google.com/vertex-ai/docs) |  | VERTEXAI_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`vertexaianthropic`](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude) |  | VERTEXAI_ANTHROPIC_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| [`vllm`](https://docs.vllm.ai/) | VLLM_API_KEY | VLLM_API_BASE | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`voyage`](https://docs.voyageai.com/) | VOYAGE_API_KEY | VOYAGE_API_BASE | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| [`watsonx`](https://www.ibm.com/watsonx) | WATSONX_API_KEY | WATSONX_URL | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ |
| [`xai`](https://x.ai/) | XAI_API_KEY | XAI_API_BASE | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| [`zai`](https://docs.z.ai/guides/develop/python/introduction) | ZAI_API_KEY | ZAI_BASE_URL | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
