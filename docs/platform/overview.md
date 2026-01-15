# Managed Platform Overview

## What is the any-llm Managed Platform?

The any-llm managed platform is a cloud-hosted service that provides secure API key vaulting and usage tracking for all your LLM providers. Instead of managing multiple provider API keys across your codebase, you get a single virtual key that works with any supported provider while keeping your credentials encrypted and your usage tracked.

The managed platform is available at [any-llm.ai](https://any-llm.ai).

## Why use the Managed Platform?

Managing LLM API keys and tracking costs across multiple providers is challenging:

- **Security risks**: API keys scattered across `.env` files, CI/CD pipelines, and developer machines
- **No visibility**: Difficult to track spending across OpenAI, Anthropic, Google, and other providers
- **Key rotation pain**: Updating keys means touching multiple systems and codebases
- **No performance insights**: No easy way to measure latency, throughput, or reliability

The managed platform solves these problems:

- **Secure Key Vault**: Your provider API keys are encrypted client-side before storage—we never see your raw keys
- **Single Virtual Key**: One `ANY_LLM_KEY` works across all providers
- **Usage Analytics**: Track tokens, costs, and performance metrics without logging prompts or responses
- **Zero Infrastructure**: No servers to deploy, no databases to manage

## How it works

The managed platform acts as a secure credential manager and usage tracker. Here's the flow:

1. **You add provider keys** to the platform dashboard (keys are encrypted in your browser before upload)
2. **You get a virtual key** (`ANY_LLM_KEY`) that represents your project
3. **Your application** uses the `PlatformProvider` with your virtual key
4. **The SDK** authenticates with the platform, retrieves and decrypts your provider key client-side
5. **Your request** goes directly to the LLM provider (OpenAI, Anthropic, etc.)
6. **Usage metadata** (tokens, model, latency) is reported back—never your prompts or responses

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Your Application                               │
│                                                                         │
│   from any_llm import completion                                        │
│   completion(provider="platform", model="openai:gpt-4", ...)            │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        any-llm SDK (PlatformProvider)                   │
│                                                                         │
│  1. Authenticate with platform using ANY_LLM_KEY                        │
│  2. Receive encrypted provider key                                      │
│  3. Decrypt provider key locally (client-side)                          │
│  4. Make request directly to provider                                   │
│  5. Report usage metadata (tokens, latency) to platform                 │
└────────────────┬─────────────────────────────────────┬──────────────────┘
                 │                                     │
                 ▼                                     ▼
┌─────────────────────────────┐       ┌────────────────────────────────────┐
│   any-llm Managed Platform  │       │        LLM Provider                │
│                             │       │   (OpenAI, Anthropic, etc.)        │
│  • Encrypted key storage    │       │                                    │
│  • Usage tracking           │       │   Your prompts/responses go        │
│  • Cost analytics           │       │   directly here—never through      │
│  • Performance metrics      │       │   our platform                     │
└─────────────────────────────┘       └────────────────────────────────────┘
```

## Key Features

### Client-Side Encryption

Your provider API keys are encrypted in your browser using XChaCha20-Poly1305 before being sent to our servers. The encryption key is derived from your account credentials and never leaves your device. This means:

- We cannot read your provider API keys
- Even if our database were compromised, your keys remain encrypted
- You maintain full control over your credentials


### Privacy-First Usage Tracking

The platform tracks usage metadata to provide cost and performance insights:

**What we track for you:**

- Token counts (input and output)
- Model name and provider
- Request timestamps
- Performance metrics (latency, throughput)

**What we never track:**

- Your prompts
- Model responses
- Any content from your conversations

### Project Organization

Organize your usage by project, team, or environment:

- Create separate projects for development, staging, and production
- Track costs per project
- Set up different provider keys per project

## Platform vs. Gateway

any-llm offers two solutions for managing LLM access. Choose the one that fits your needs:

| Feature | Managed Platform | Self-Hosted Gateway |
|---------|-----------------|---------------------|
| **Deployment** | Cloud-hosted (no infrastructure) | Self-hosted (Docker + Postgres) |
| **Key Storage** | Client-side encrypted vault | Your own configuration |
| **Budget Enforcement** | Coming soon | Built-in |
| **User Management** | Per-project | Full user/key management |
| **Request Routing** | Direct to provider, no proxy | Through your gateway |
| **Best For** | Teams wanting zero-ops key management and usage tracking| Organizations needing full control |

You can also use both together—store your provider keys in the managed platform and use them in a self-hosted gateway deployment.

## Current Status

The any-llm managed platform is in **open beta**. During the beta:

- **Free access** to all features
- Core encryption and key management are **production-ready**
- Dashboard UX and advanced features are being refined
- Feedback is welcome at [any-llm.ai](https://any-llm.ai)

## Getting Started

Ready to try the managed platform?

1. Create an account at [any-llm.ai](https://any-llm.ai)
2. Add your provider API keys
3. Get your virtual key
4. Make your first request
