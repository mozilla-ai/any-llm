"""Example demonstrating httpx client usage for connection pooling with OpenAI providers."""
# ruff: noqa: T201

import asyncio

import httpx

from any_llm import acompletion, completion


def sync_example():
    """Demonstrate httpx client usage for sync completion calls."""
    print("=== Sync Example ===")

    # Configure httpx client with connection pooling
    with httpx.Client(
        timeout=httpx.Timeout(30.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        http2=True,  # Enable HTTP/2 where supported
    ) as http_client:
        # Make multiple requests using the same connection pool
        for i in range(3):
            try:
                response = completion(
                    model="gpt-3.5-turbo",
                    provider="openai",
                    messages=[{"role": "user", "content": f"Say hello #{i + 1} in a creative way!"}],
                    http_client=http_client,
                )
                print(f"Response {i + 1}: {response.choices[0].message.content}")
            except Exception as e:
                print(f"Error in request {i + 1}: {e}")


async def async_example():
    """Demonstrate httpx client usage for async completion calls."""
    print("\n=== Async Example ===")

    # Configure async httpx client
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        http2=True,
    ) as http_client:
        # Create multiple concurrent tasks
        tasks = []
        for i in range(3):
            task = acompletion(
                model="gpt-3.5-turbo",
                provider="openai",
                messages=[{"role": "user", "content": f"Tell me a short joke #{i + 1}"}],
                http_client=http_client,
            )
            tasks.append(task)

        # Execute all tasks concurrently
        try:
            responses = await asyncio.gather(*tasks)
            for i, response in enumerate(responses):
                print(f"Joke {i + 1}: {response.choices[0].message.content}")
        except Exception as e:
            print(f"Error in async requests: {e}")


def different_providers_example():
    """Show httpx client compatibility with BaseOpenAI providers."""
    print("\n=== Different Providers Example ===")

    providers_to_test = [
        ("openai", "gpt-3.5-turbo"),
        # Add your preferred providers here
        # ("databricks", "databricks-meta-llama-3-1-8b-instruct"),
        # ("deepseek", "deepseek-chat"),
    ]

    with httpx.Client(timeout=httpx.Timeout(15.0)) as http_client:
        for provider, model in providers_to_test:
            try:
                response = completion(
                    model=model,
                    provider=provider,
                    messages=[{"role": "user", "content": "What is 2+2?"}],
                    http_client=http_client,
                )
                print(f"{provider.upper()}: {response.choices[0].message.content}")
            except Exception as e:
                print(f"Error with {provider}: {e}")


if __name__ == "__main__":
    print("OpenAI Providers httpx Client Example")
    print("=" * 40)

    # Run sync example
    sync_example()

    # Run async example
    asyncio.run(async_example())

    # Run different providers example
    different_providers_example()

    print("\nDone! Connection pooling helps reduce overhead when making multiple requests.")
