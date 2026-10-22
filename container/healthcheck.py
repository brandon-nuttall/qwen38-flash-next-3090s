# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)

"""Require a real one-token completion from a model advertised by this server."""

import json
import os
import sys
import urllib.request

base = "http://127.0.0.1:" + os.environ.get("PORT", "8000")
headers = {"Content-Type": "application/json"}
if os.environ.get("VLLM_API_KEY"):
    headers["Authorization"] = "Bearer " + os.environ["VLLM_API_KEY"]


def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=40) as response:
        if response.status != 200:
            raise ValueError(f"{path} returned HTTP {response.status}, expected 200")
        return json.load(response)


try:
    model = request("/v1/models")["data"][0]["id"]
    result = request(
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 1,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    if not result.get("choices"):
        raise ValueError("generation returned no choices")
    tokens = result.get("usage", {}).get("completion_tokens")
    if type(tokens) is not int or tokens < 1:
        raise ValueError("generation returned no generated tokens")
except Exception as exc:
    sys.exit(f"healthcheck failed: {type(exc).__name__}: {exc}")
