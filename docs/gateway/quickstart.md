# Quick Start

This guide will help you set up any-llm-gateway and make your first LLM completion request. The gateway acts as a proxy between your applications and LLM providers, providing cost control, usage tracking, and API key management.

By the end of this guide, you will:  

1. Configure provider credentials and model pricing (e.g., OpenAI API key)  
1. Run the gateway   
1. Authenticate requests using a master key  
1. Make completion requests through the gateway  

> **Note:** for the purposes of this quickstart we will utilize the docker-compose and config.yml file, but alternative configuration designs are available and detailed [here](./configuration.md)

## Pre-Requisites

1. Docker
1. Access to at least one LLM provider

## Configure and run the Gateway

When running any-llm-gateway, it must have a few things configured:

1. `GATEWAY_MASTER_KEY`. This master key has admin access to manage budgets, users, virtual keys, etc.
1. `DATABASE_URL`. The gateway relies upon a postgres database for storage.
1. Provider Keys. The gateway connects to providers (Mistral, AWS, Vertex, Azure, etc) using credentials that must be set.

For the purposes of the quickstart we will use the included `docker/docker-compose.yml`, but you can customize the file to your own requirements.

### Generate  master key

First, generate a secure master key: 
```python 
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Save the output of this command, you'll need it in the next steps. 

### Configure providers 

Copy the example `docker/config.example.yml` file to `docker/config.yml`

```bash
cp docker/config.example.yml docker/config.yml
```

At a minimum you'll need to fill out the value for the master_key, and also enter credential information for at least one provider. You can browse supported providers [here](https://mozilla-ai.github.io/any-llm/providers/). If you would like to track usage cost, you'll also need to configure model pricing, as explained in the config template file.

```yaml
database_url: "postgresql://gateway:gateway@postgres:5432/gateway"

master_key: 09kS0xTiz6JqO....

providers:
  openai:
    api_key: YOUR_OPENAI_API_KEY_HERE
    api_base: "https://api.openai.com/v1"  # optional, useful when you want to use a specific version of the API

models:
  openai:gpt-4:
    input_price_per_million: 0.15
    output_price_per_million: 0.60
```


### Start the gateway

Run the docker-compose file, ensuring that your config.yml is located in the same directory as the docker-compose.yml file (`docker/config.yml`).

The default setting is to build the gateway from source, but see the docker-compose.yml file comment to see how to use a published version of any-llm-gateway instead of the source code.

```bash
# From project root directory
docker-compose -f docker/docker-compose.yml up -d --build
```

````bash
# Verify the gateway is running
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy"}
````

### View Logs 

```bash
docker-compose -f docker/docker-compose.yml logs -f
```


## Create a user and make your first request

Now that it's running, clients can make requests! The gateway supports two authentication patterns: use of the master key, or virtual keys. See the [authentication doc](./authentication.md) for more information. For this guide we will use the master key for both administration and client requests.

To make the below commands easier to run, you can set the key as an env var in your terminal:

```bash
export GATEWAY_MASTER_KEY=YOUR_MASTER_KEY
```

> **tip**: for the below `curl` commands, append `| jq` in order for it be pretty-printed in the console.

### Create a user

In order to track usage, we must first create a user so that we can associate our completion request with them.

```bash
curl -s -X POST http://localhost:8000/v1/users \
  -H "X-AnyLLM-Key: Bearer ${GATEWAY_MASTER_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user-123", "alias": "Bob"}'
```

<details>
<summary>Sample Response</summary>

```bash
{
    "user_id": "user-123",
    "alias": "Bob",
    "spend": 0,
    "budget_id": null,
    "budget_started_at": null,
    "next_budget_reset_at": null,
    "blocked": false,
    "created_at": "2025-11-07T16:41:44.429258+00:00",
    "updated_at": "2025-11-07T16:41:44.429261+00:00",
    "metadata": {}
}
```
</details>

### Make a request

Make a completion request using the master key and specify that the completion should be attached to the user you just created. This is only required when authenticating using the master key, if a user has a virtual key they do not need to specify a user id. You may also need to adjust the model to match one of the providers that you configured when running the gateway.

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "X-AnyLLM-Key: Bearer ${GATEWAY_MASTER_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4",
    "messages": [{"role": "user", "content": "Write a haiku on Uranus!"}],
    "user": "user-123"
  }'
```

<details>  
<summary>Sample Response </summary>

```json 
{
    "id": "chatcmpl-CZJvdiwHSdCZ2TfIPhutgPY4WYP46",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "logprobs": null,
            "message": {
                "content": "Gleaming ice-blue sphere,\nCircling in celestial dance,\nUranus, so clear.",
                "refusal": null,
                "role": "assistant",
                "annotations": [],
                "audio": null,
                "function_call": null,
                "tool_calls": null,
                "reasoning": null
            }
        }
    ],
    "created": 1762534121,
    "model": "gpt-4-0613",
    "object": "chat.completion",
    "service_tier": "default",
    "system_fingerprint": null,
    "usage": {
        "completion_tokens": 21,
        "prompt_tokens": 15,
        "total_tokens": 36,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 0,
            "rejected_prediction_tokens": 0
        },
        "prompt_tokens_details": {
            "audio_tokens": 0,
            "cached_tokens": 0
        }
    }
}
```

</details>

Alternatively, if you are using the any-llm python sdk, you can access using the gateway client.

```python
import os
from any_llm import completion

completion(
  provider="gateway",
  model="openai:gpt-4",
  api_base="http://localhost:8000/v1",
  api_key=os.environ['GATEWAY_MASTER_KEY'],
  messages=[{"role": "user", "content": "Write a haiku on Uranus!"}],
  user="user-123",
)
```

### View metrics

Now using the master key, we can access the usage information for the user.

```bash
curl -s http://localhost:8000/v1/users/user-123 \
  -H "X-AnyLLM-Key: Bearer ${GATEWAY_MASTER_KEY}" \
  -H "Content-Type: application/json"
```

<details>
<summary>Sample Response </summary>

```json
{
    "user_id": "user-123",
    "alias": "Bob",
    "spend": 0.0000216,
    "budget_id": null,
    "budget_started_at": null,
    "next_budget_reset_at": null,
    "blocked": false,
    "created_at": "2025-11-07T16:41:44.429258+00:00",
    "updated_at": "2025-11-07T16:48:42.972327+00:00",
    "metadata": {}
}
```
</details>  

You'll notice that the user does not have a budget attached, which means that we track their usage but do not limit them! For more information on creating and managing budgets and budget reset cycles, see the [Budget Management docs](budget-management.md)

## Next Steps



- **[Configuration](configuration.md)** - Configure providers, pricing, and other settings
- **[Authentication](authentication.md)** - Learn about master keys and virtual API keys
- **[Budget Management](budget-management.md)** - Set spending limits and track costs
- **[API Reference](api-reference.md)** - Explore the complete API
