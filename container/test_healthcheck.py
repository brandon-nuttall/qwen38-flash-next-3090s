# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)

"""Exercise readiness through fake HTTP responses without a server or GPU."""

import io
import json
import os
import runpy
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("healthcheck.py")


class Response(io.BytesIO):
    def __init__(self, body, status=200):
        super().__init__(json.dumps(body).encode())
        self.status = status


class HealthcheckTests(unittest.TestCase):
    def run_check(self, completion, *, status=200, listing_status=200, error=None):
        replies = [
            Response({"data": [{"id": "test-model"}]}, listing_status),
            error if error else Response(completion, status),
        ]
        with patch.dict(os.environ, {"VLLM_API_KEY": "test-key"}, clear=True):
            with patch("urllib.request.urlopen", side_effect=replies) as opened:
                runpy.run_path(str(SCRIPT), run_name="__main__")
                request = opened.call_args.args[0]
                self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
                self.assertEqual(json.loads(request.data)["model"], "test-model")

    def test_completed_token_is_ready(self):
        self.run_check({"choices": [{}], "usage": {"completion_tokens": 1}})

    def test_empty_or_invalid_token_usage_is_unready(self):
        for tokens in (0, None, -1, True, "1"):
            with self.subTest(tokens=tokens), self.assertRaises(SystemExit):
                self.run_check(
                    {"choices": [{}], "usage": {"completion_tokens": tokens}}
                )

    def test_missing_choice_is_unready(self):
        with self.assertRaises(SystemExit):
            self.run_check({"choices": [], "usage": {"completion_tokens": 1}})

    def test_success_json_with_non_200_status_is_unready(self):
        with self.assertRaises(SystemExit):
            self.run_check(
                {"choices": [{}], "usage": {"completion_tokens": 1}}, status=201
            )

    def test_model_listing_requires_http_200(self):
        with self.assertRaises(SystemExit):
            self.run_check({}, listing_status=201)

    def test_authentication_failure_is_unready(self):
        with self.assertRaises(SystemExit):
            self.run_check(
                {},
                error=urllib.error.HTTPError(
                    "http://test", 401, "Unauthorized", {}, None
                ),
            )


if __name__ == "__main__":
    unittest.main()
