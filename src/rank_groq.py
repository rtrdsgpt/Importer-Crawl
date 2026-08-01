"""
Phase 3 (Groq API variant).

Free tier, fast inference, and serves much larger open models (e.g.
Llama-3.3-70B) than what's available on Hugging Face's free tier -- a good
free fallback when OpenAI/Anthropic credits aren't available. Groq exposes
an OpenAI-compatible endpoint, so this reuses the `openai` SDK pointed at
Groq's base URL.

Usage:
    export GROQ_API_KEY=gsk_...
    python src/rank_groq.py --input data/scraped_ceramic_tiles_germany.json \\
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
from openai import APIConnectionError, APIError, OpenAI, RateLimitError
from pydantic import ValidationError

import rank_common as rc

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def judge_page(
    client: OpenAI, model: str, page: dict, product: str, country: str,
    retries: int = 2,
) -> rc.LLMJudgment | None:
    messages = [
        {"role": "system", "content": rc.SYSTEM_PROMPT},
        {"role": "user", "content": rc.build_prompt(page, product, country)},
    ]

    last_error = None
    for attempt in range(1, retries + 2):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, max_tokens=500, temperature=0.1,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            data = rc.extract_json_object(content)
            return rc.LLMJudgment.model_validate(data)
        except RateLimitError as exc:
            last_error = exc
            wait = 2 ** attempt
            print(f"  ! rate limited ({exc}); retrying in {wait}s...")
            time.sleep(wait)
        except (APIConnectionError, APIError) as exc:
            last_error = exc
            wait = 2 ** attempt
            print(f"  ! Groq API error ({exc}); retrying in {wait}s...")
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
    parser = argparse.ArgumentParser(description="Rank scraped companies via the Groq API.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--product", required=True)
    parser.add_argument("--country", required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-score", type=int, default=40)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--groq-key", default=None, help="Defaults to GROQ_API_KEY env var")
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    api_key = args.groq_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("No Groq API key found. Set GROQ_API_KEY in your environment or .env "
                          "(free key: https://console.groq.com/keys).")

    input_path = Path(args.input)
    pages = json.loads(input_path.read_text(encoding="utf-8"))
    output_path = rc.default_output_path(input_path, args.output)

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    judge_fn = partial(judge_page, client, args.model)

    ranked = rc.rank_companies(
        pages, product=args.product, country=args.country, judge_fn=judge_fn,
        top_n=args.top_n, min_score=args.min_score, delay_seconds=args.delay,
    )
    rc.save_ranked(ranked, output_path)
    rc.print_summary(ranked, args.min_score, output_path)


if __name__ == "__main__":
    main()
