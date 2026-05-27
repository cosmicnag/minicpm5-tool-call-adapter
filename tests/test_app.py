from pathlib import Path
import json
import sys
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minicpm_tool_call_adapter.app import (
    _response_headers,
    _sse_stream_from_completion,
    rewrite_completion_response,
)
from minicpm_tool_call_adapter.parser import MiniCPM5Parser


class AppRewriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = MiniCPM5Parser()
        self.tools = [
            {
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "days": {"type": "integer"},
                        },
                        "required": ["city"],
                    },
                }
            }
        ]

    def test_rewrite_completion_response_converts_xml_to_tool_calls(self) -> None:
        data = {
            "id": "chatcmpl-test",
            "model": "minicpm5-1b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": (
                            'hello <function name="get_weather">'
                            '<param name="city">Paris</param>'
                            '<param name="days">2</param>'
                            "</function>"
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        rewritten = rewrite_completion_response(data, self.tools, self.parser)
        message = rewritten["choices"][0]["message"]

        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["content"], "hello ")
        self.assertEqual(message["tool_calls"][0]["id"], "chatcmpl-test-call-0-0")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(
            json.loads(message["tool_calls"][0]["function"]["arguments"]),
            {"city": "Paris", "days": 2},
        )
        self.assertEqual(rewritten["choices"][0]["finish_reason"], "tool_calls")

    def test_rewrite_completion_response_leaves_plain_text_alone(self) -> None:
        data = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "plain answer"},
                    "finish_reason": "stop",
                }
            ],
        }

        rewritten = rewrite_completion_response(data, self.tools, self.parser)

        self.assertEqual(rewritten["choices"][0]["message"]["content"], "plain answer")
        self.assertNotIn("tool_calls", rewritten["choices"][0]["message"])
        self.assertEqual(rewritten["choices"][0]["finish_reason"], "stop")


class ResponseHelpersTests(unittest.TestCase):
    def test_response_headers_drop_hop_by_hop_headers(self) -> None:
        headers = {
            "content-type": "application/json",
            "content-length": "123",
            "connection": "keep-alive",
            "x-custom": "ok",
        }

        filtered = _response_headers(headers)

        self.assertEqual(filtered, {"content-type": "application/json", "x-custom": "ok"})


class SSEStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_streaming_completion_emits_done_and_tool_calls(self) -> None:
        data = {
            "id": "chatcmpl-test",
            "model": "minicpm5-1b",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hello",
                        "tool_calls": [
                            {
                                "id": "chatcmpl-test-call-0-0",
                                "type": "function",
                                "index": 0,
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Paris"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        chunks = [chunk async for chunk in _sse_stream_from_completion(data)]

        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        payloads = [json.loads(chunk.removeprefix("data: ").strip()) for chunk in chunks[:-1]]
        self.assertEqual(payloads[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual(payloads[1]["choices"][0]["delta"], {"content": "hello"})
        self.assertEqual(payloads[2]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(payloads[-1]["choices"][0]["finish_reason"], "tool_calls")
