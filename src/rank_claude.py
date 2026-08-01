"""
Phase 3 (Anthropic Claude API variant).

Recommended default for accuracy: the free Hugging Face 7B model (rank_hf.py)
was observed confidently misclassifying manufacturers as buyers -- Claude
Sonnet 5 handles the buyer-vs-seller distinction this task depends on far
more reliably. Use --model claude-haiku-4-5 for a cheaper/faster run if
budget matters more than accuracy.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python src/rank_claude.py --input data/scraped_ceramic_tiles_germany.json \\
        --product "Ceramic Tiles" --country "Germany" --top-n 10
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

from anthropic import Anthropic, APIConnectionError, APIError, RateLimitError
from dotenv import load_dotenv
from pydantic import ValidationError

import rank_common as rc

load_dotenv()

DEFAULT_MODEL = "claude-sonnet-5"


def judge_page(
    client: Anthropic, model: str, page: dict, product: str, country: str,
    retries: int = 2,
) -> rc.LLMJudgment | None:
    messages = [{"role": "user", "content": rc.build_prompt(page, product, country)}]

    last_error = None
    for attempt in range(1, retries + 2):
        try:
            response = client.messages.create(
                model=model, max_tokens=500, system=rc.SYSTEM_PROMPT, messages=messages,
            )
            if response.stop_reason == "refusal":
                print(f"  ! model refused to judge {page['url']}; skipping")
                return None
            content = next(b.text for b in response.content if b.type == "text")
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
            print(f"  ! Claude API error ({exc}); retrying in {wait}s...")
            time.sleep(wait)
        except (ValueError, StopIteration, json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            messages.append({"role": "assistant", "content": "(invalid JSON response)"})
            messages.append({"role": "user", "content": (
                "Your previous response was not valid JSON matching the schema. "
                "Respond with ONLY the JSON object, nothing else."
            )})
            print(f"  ! bad output from model ({exc}); retrying...")

    print(f"  ! giving up on {page['url']}: {last_error}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank scraped companies via the Claude API.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--product", required=True)
    parser.add_argument("--country", required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-score", type=int, default=40)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                         help="e.g. claude-sonnet-5 (default) or claude-haiku-4-5 for a cheaper run")
    parser.add_argument("--anthropic-key", default=None, help="Defaults to ANTHROPIC_API_KEY env var")
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    api_key = args.anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("No Anthropic API key found. Set ANTHROPIC_API_KEY in your environment or .env.")

    input_path = Path(args.input)
    pages = json.loads(input_path.read_text(encoding="utf-8"))
    output_path = rc.default_output_path(input_path, args.output)

    client = Anthropic(api_key=api_key)
    judge_fn = partial(judge_page, client, args.model)

    ranked = rc.rank_companies(
        pages, product=args.product, country=args.country, judge_fn=judge_fn,
        top_n=args.top_n, min_score=args.min_score, delay_seconds=args.delay,
    )
    rc.save_ranked(ranked, output_path)
    rc.print_summary(ranked, args.min_score, output_path)


if __name__ == "__main__":
    main()
