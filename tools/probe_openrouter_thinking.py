"""Does a run config's model really think (or really not think) on OpenRouter?

One short request with exactly the model, sampling and `extra` the config
sends, then a verdict on the reply:

    on   -- native reasoning in its own field AND a non-empty answer
    off  -- no native reasoning at all, a non-empty answer, no inline thinking
    anything else is printed as the reason, and the exit code is 1.

    .venv/bin/python tools/probe_openrouter_thinking.py <run config> on|off
"""

from __future__ import annotations

import json
import sys
import urllib.request

from abductionbench.core.config import load_run_config

QUESTION = (
    "A glass of water left in a warm room is empty after three days, and nobody "
    "drank from it. Which is more likely: 1) evaporation 2) a leak? Give the "
    "label inside <answer></answer>."
)


def main() -> int:
    config_path, expected = sys.argv[1], sys.argv[2]
    model = load_run_config(config_path).models[0]
    payload = {
        "model": model.model_name,
        "messages": [{"role": "user", "content": QUESTION}],
        "max_tokens": 4000,
        "temperature": 0.7,
        "top_p": model.sampling.top_p,
        "seed": model.sampling.seed,
        **model.sampling.extra,
    }
    request = urllib.request.Request(
        f"{model.endpoint.base_url}{model.endpoint.chat_path}",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {model.endpoint.api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.loads(response.read())
    except Exception as exc:  # noqa: BLE001
        print(f"no reply ({type(exc).__name__}: {str(exc)[:200]})")
        return 1
    if "error" in data or not data.get("choices"):
        print(f"error reply: {str(data.get('error') or data)[:300]}")
        return 1
    message = data["choices"][0].get("message") or {}
    reasoning = (message.get("reasoning") or message.get("reasoning_content") or "").strip()
    content = (message.get("content") or "").strip()
    inline = any(tag in content for tag in ("<think>", "</think>", "Thinking Process"))
    where = f"provider={data.get('provider')} extra={json.dumps(model.sampling.extra)}"
    if not content:
        verdict = "no answer (content empty)"
    elif reasoning:
        verdict = "on"
    elif inline:
        verdict = "inline thinking in content"
    else:
        verdict = "off"
    print(f"{verdict} ({where}, reasoning {len(reasoning)} chars, content {len(content)} chars)")
    return 0 if verdict == expected else 1


if __name__ == "__main__":
    sys.exit(main())
