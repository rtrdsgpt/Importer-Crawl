"""
Phase 3 (Google Gemini API variant).

Free tier (generous daily quota on Gemini Flash models). Gemini exposes an
OpenAI-compatible endpoint, so this reuses the `openai` SDK pointed at
Google's base URL rather than pulling in the separate google-genai SDK.

Usage:
    export GEMINI_API_KEY=...
    python src/rank_gemini.py --input data/scraped_ceramic_tiles_germany.json \\
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

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-3.6-flash"


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
                # Gemini 2.5+/3.x models "think" before answering by default; a
                # small max_tokens can be entirely consumed by hidden reasoning
                # tokens, leaving no room for the visible JSON output.
                model=model, messages=messages, max_tokens=2048, temperature=0.1,
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
            print(f"  ! Gemini API error ({exc}); retrying in {wait}s...")
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
    parser = argparse.ArgumentParser(description="Rank scraped companies via the Gemini API.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--product", required=True)
    parser.add_argument("--country", required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-score", type=int, default=40)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gemini-key", default=None, help="Defaults to GEMINI_API_KEY env var")
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    api_key = args.gemini_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("No Gemini API key found. Set GEMINI_API_KEY in your environment or .env "
                          "(free key: https://aistudio.google.com/apikey).")

    input_path = Path(args.input)
    pages = json.loads(input_path.read_text(encoding="utf-8"))
    output_path = rc.default_output_path(input_path, args.output)

    client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
    judge_fn = partial(judge_page, client, args.model)

    ranked = rc.rank_companies(
        pages, product=args.product, country=args.country, judge_fn=judge_fn,
        top_n=args.top_n, min_score=args.min_score, delay_seconds=args.delay,
    )
    rc.save_ranked(ranked, output_path)
    rc.print_summary(ranked, args.min_score, output_path)


if __name__ == "__main__":
    main()
