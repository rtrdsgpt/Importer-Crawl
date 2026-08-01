"""
Phase 3 (Hugging Face Inference API variant).

Free/cheap option. Note: the free tier's included monthly credits are
small and smaller open models are noticeably weaker at the buyer-vs-seller
distinction this task depends on (see README for the Agrob Buchtal
example) -- prefer rank_openai.py or rank_claude.py when accuracy matters
more than cost.

Usage:
    export HF_TOKEN=hf_...
    python src/rank_hf.py --input data/scraped_ceramic_tiles_germany.json \\
        --product "Ceramic Tiles" --country "Germany" --top-n 10
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from pydantic import ValidationError

import rank_common as rc

load_dotenv()

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def judge_page(
    client: InferenceClient, model: str, page: dict, product: str, country: str,
    retries: int = 2,
) -> rc.LLMJudgment | None:
    messages = [
        {"role": "system", "content": rc.SYSTEM_PROMPT},
        {"role": "user", "content": rc.build_prompt(page, product, country)},
    ]

    last_error = None
    for attempt in range(1, retries + 2):
        try:
            response = client.chat_completion(
                messages=messages, model=model, max_tokens=500, temperature=0.1,
            )
            content = response.choices[0].message.content
            data = rc.extract_json_object(content)
            return rc.LLMJudgment.model_validate(data)
        except HfHubHTTPError as exc:
            last_error = exc
            wait = 2 ** attempt
            print(f"  ! HF API error ({exc}); retrying in {wait}s...")
            time.sleep(wait)
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            messages.append({"role": "user", "content": (
                "Your previous response was not valid JSON matching the schema. "
                "Respond with ONLY the JSON object, nothing else."
            )})
            print(f"  ! bad output from model ({exc}); retrying...")

    print(f"  ! giving up on {page['url']}: {last_error}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank scraped companies via Hugging Face Inference API.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--product", required=True)
    parser.add_argument("--country", required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-score", type=int, default=40)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--hf-token", default=None, help="Defaults to HF_TOKEN env var")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit("No Hugging Face token found. Set HF_TOKEN in your environment or .env.")

    input_path = Path(args.input)
    pages = json.loads(input_path.read_text(encoding="utf-8"))
    output_path = rc.default_output_path(input_path, args.output)

    client = InferenceClient(model=args.model, token=hf_token)
    judge_fn = partial(judge_page, client, args.model)

    ranked = rc.rank_companies(
        pages, product=args.product, country=args.country, judge_fn=judge_fn,
        top_n=args.top_n, min_score=args.min_score, delay_seconds=args.delay,
    )
    rc.save_ranked(ranked, output_path)
    rc.print_summary(ranked, args.min_score, output_path)


if __name__ == "__main__":
    main()
