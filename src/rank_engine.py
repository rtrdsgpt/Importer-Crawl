"""
Phase 3: LLM Reasoning, Filtering & Ranking.

Single entry point for every supported LLM provider -- pick one with
--provider. Groq, Gemini, and OpenAI all speak the OpenAI-compatible
chat-completions format, so they share one client-construction + retry
path (judge_openai_compatible); Claude and Hugging Face each need their
own SDK and get their own judge function. Every provider shares the same
prompt, schema, and hallucination-guard logic in rank_schema.py.

Usage:
    export GROQ_API_KEY=gsk_...   # or OPENAI_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY / HF_TOKEN
    python src/rank_engine.py --input data/scraped_ceramic_tiles_germany.json \\
        --product "Ceramic Tiles" --country "Germany" --provider groq
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

import rank_schema as rc

load_dotenv()

PROVIDER_CONFIGS = {
    "groq": {
        "kind": "openai_compatible", "env_var": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile", "max_tokens": 500,
    },
    "gemini": {
        "kind": "openai_compatible", "env_var": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-3.6-flash",
        # Gemini 2.5+/3.x models "think" before answering by default; a
        # small max_tokens can be entirely consumed by hidden reasoning
        # tokens, leaving no room for the visible JSON output.
        "max_tokens": 2048,
    },
    "openai": {
        "kind": "openai_compatible", "env_var": "OPENAI_API_KEY",
        "base_url": None, "default_model": "gpt-4o-mini", "max_tokens": 500,
    },
    "claude": {
        "kind": "anthropic", "env_var": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-5", "max_tokens": 500,
    },
    "hf": {
        "kind": "hf", "env_var": "HF_TOKEN",
        "default_model": "Qwen/Qwen2.5-7B-Instruct", "max_tokens": 500,
    },
}


def judge_openai_compatible(
    client, model: str, max_tokens: int, page: dict, product: str, country: str,
    retries: int = 2,
) -> rc.LLMJudgment | None:
    from openai import APIConnectionError, APIError, RateLimitError

    messages = [
        {"role": "system", "content": rc.SYSTEM_PROMPT},
        {"role": "user", "content": rc.build_prompt(page, product, country)},
    ]

    last_error = None
    for attempt in range(1, retries + 2):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=0.1,
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
            print(f"  ! API error ({exc}); retrying in {wait}s...")
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


def judge_claude(
    client, model: str, max_tokens: int, page: dict, product: str, country: str,
    retries: int = 2,
) -> rc.LLMJudgment | None:
    from anthropic import APIConnectionError, APIError, RateLimitError

    messages = [{"role": "user", "content": rc.build_prompt(page, product, country)}]

    last_error = None
    for attempt in range(1, retries + 2):
        try:
            response = client.messages.create(
                model=model, max_tokens=max_tokens, system=rc.SYSTEM_PROMPT, messages=messages,
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


def judge_hf(
    client, model: str, max_tokens: int, page: dict, product: str, country: str,
    retries: int = 2,
) -> rc.LLMJudgment | None:
    from huggingface_hub.errors import HfHubHTTPError

    messages = [
        {"role": "system", "content": rc.SYSTEM_PROMPT},
        {"role": "user", "content": rc.build_prompt(page, product, country)},
    ]

    last_error = None
    for attempt in range(1, retries + 2):
        try:
            response = client.chat_completion(
                messages=messages, model=model, max_tokens=max_tokens, temperature=0.1,
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


def build_judge_fn(provider: str, model: str, api_key: str):
    """Returns a judge_fn(page, product, country) -> LLMJudgment | None for
    the given provider, with its client already constructed."""
    config = PROVIDER_CONFIGS[provider]
    max_tokens = config["max_tokens"]

    if config["kind"] == "openai_compatible":
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=config["base_url"])
        return partial(judge_openai_compatible, client, model, max_tokens)

    if config["kind"] == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        return partial(judge_claude, client, model, max_tokens)

    if config["kind"] == "hf":
        from huggingface_hub import InferenceClient
        client = InferenceClient(model=model, token=api_key)
        return partial(judge_hf, client, model, max_tokens)

    raise ValueError(f"unknown provider kind: {config['kind']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank scraped companies by relevance via an LLM.")
    parser.add_argument("--input", required=True, help="Path to scraped_*.json from scraper.py")
    parser.add_argument("--product", required=True, help='e.g. "Ceramic Tiles"')
    parser.add_argument("--country", required=True, help='e.g. "Germany"')
    parser.add_argument("--provider", required=True, choices=list(PROVIDER_CONFIGS),
                         help="Which LLM provider to rank with.")
    parser.add_argument("--top-n", type=int, default=10, help="Max companies to keep")
    parser.add_argument("--min-score", type=int, default=40, help="Minimum relevance score to keep")
    parser.add_argument("--model", default=None, help="Defaults to the provider's default model")
    parser.add_argument("--api-key", default=None,
                         help="Defaults to the provider's env var (see --help per provider below)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between LLM calls (s)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    config = PROVIDER_CONFIGS[args.provider]
    model = args.model or config["default_model"]
    api_key = args.api_key or os.environ.get(config["env_var"])
    if not api_key:
        raise SystemExit(
            f"No API key found for provider {args.provider!r}. "
            f"Set {config['env_var']} in your environment or .env, or pass --api-key."
        )

    input_path = Path(args.input)
    pages = json.loads(input_path.read_text(encoding="utf-8"))
    output_path = rc.default_output_path(input_path, args.output)

    judge_fn = build_judge_fn(args.provider, model, api_key)

    ranked = rc.rank_companies(
        pages, product=args.product, country=args.country, judge_fn=judge_fn,
        top_n=args.top_n, min_score=args.min_score, delay_seconds=args.delay,
    )
    rc.save_ranked(ranked, output_path)
    rc.print_summary(ranked, args.min_score, output_path)


if __name__ == "__main__":
    main()
