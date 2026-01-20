"""Example integration with any-llm Python client."""

import asyncio

from any_llm import AnyLLM


async def test_with_any_llm():
    """Run acceptance tests using the any-llm Python client."""

    llm = AnyLLM.create(
        "openai",
        api_key="test-key",
        api_base="http://localhost:8080/v1",
    )

    print("Testing basic completion with any-llm...")
    result = await llm.acompletion(
        model="test-basic",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(f"Response: {result.choices[0].message.content}")
    print()

    print("Testing tool calls with any-llm...")

    def get_weather(location: str) -> str:
        """Get the weather for a location.

        Args:
            location: The city name.
        """
        return f"Weather in {location}: sunny, 20C"

    result = await llm.acompletion(
        model="test-tools",
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=[get_weather],
    )
    print(f"Tool calls: {result.choices[0].message.tool_calls}")
    print()

    print("Testing streaming with any-llm...")
    print("Response: ", end="")
    async for chunk in await llm.acompletion(
        model="test-stream",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    ):
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
    print("\n")


if __name__ == "__main__":
    asyncio.run(test_with_any_llm())
