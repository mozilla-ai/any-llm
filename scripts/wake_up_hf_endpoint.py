import argparse
import time

from huggingface_hub.errors import HfHubHTTPError

from any_llm import completion

HF_ENDPOINT = "https://y0okp71n85ezo5nr.us-east-1.aws.endpoints.huggingface.cloud/v1/"


def wake_up_hf_endpoint(retry: int = 0):
    while True:
        try:
            completion(
                model="huggingface:tgi", messages=[{"role": "user", "content": "Are you awake?"}], api_base=HF_ENDPOINT
            )
            break
        except HfHubHTTPError as e:
            if not retry:
                print(f"Endpoint not ready, giving up...\n{e}")
                return

            print(f"Endpoint not ready, retrying...\n{e}")
            time.sleep(retry)

    print("Endpoint ready")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wake up Hugging Face endpoint")
    parser.add_argument("--retry", type=int, default=0, help="Retry interval in seconds (0 means no retry)")
    args = parser.parse_args()
    wake_up_hf_endpoint(retry=args.retry)
