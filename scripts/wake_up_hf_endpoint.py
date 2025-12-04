import argparse
import time

from aiohttp.client_exceptions import ClientResponseError
from huggingface_hub.errors import HfHubHTTPError

from any_llm.api import completion

HF_ENDPOINT = "https://oze7k8n86bjfzgjk.us-east-1.aws.endpoints.huggingface.cloud/v1"


def wake_up_hf_endpoint(retry_count: int = 0, retry_interval: int = 10):
    attempt = 0
    max_attempts = retry_count + 1

    while attempt < max_attempts:
        try:
            completion(
                model="huggingface:tgi", messages=[{"role": "user", "content": "Are you awake?"}], api_base=HF_ENDPOINT
            )
        except (ClientResponseError, HfHubHTTPError) as e:
            attempt += 1
            if attempt >= max_attempts:
                print(f"Endpoint not ready after {attempt} attempts, giving up...\n{e}")
                return

            print(f"Endpoint not ready (attempt {attempt}/{max_attempts}), retrying in {retry_interval}s...\n{e}")
            time.sleep(retry_interval)
        else:
            print("Endpoint ready")
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wake up Hugging Face endpoint")
    parser.add_argument("--retry-count", type=int, default=0, help="Number of retry attempts (default: 0, no retries)")
    parser.add_argument("--retry-interval", type=int, default=10, help="Seconds between retry attempts (default: 10)")
    args = parser.parse_args()
    wake_up_hf_endpoint(retry_count=args.retry_count, retry_interval=args.retry_interval)
