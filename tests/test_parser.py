from pathlib import Path
import json
import sys
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minicpm_tool_call_adapter.parser import (
    MiniCPM5Parser,
    ParsedToolCall,
    build_openai_tool_calls,
    build_tool_specs,
    parse_arguments,
)


class ParseArgumentsTests(unittest.TestCase):
    def test_parse_arguments_handles_json_and_python_literals(self) -> None:
        parsed_json, ok_json = parse_arguments('{"city": "Paris", "days": 2}')
        parsed_py, ok_py = parse_arguments("{'city': 'Paris', 'days': 2}")
        parsed_text, ok_text = parse_arguments("plain text")

        self.assertTrue(ok_json)
        self.assertEqual(parsed_json, {"city": "Paris", "days": 2})
        self.assertTrue(ok_py)
        self.assertEqual(parsed_py, {"city": "Paris", "days": 2})
        self.assertFalse(ok_text)
        self.assertEqual(parsed_text, "plain text")


class ToolSpecTests(unittest.TestCase):
    def test_build_tool_specs_extracts_allowed_required_and_types(self) -> None:
        specs = build_tool_specs(
            [
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
        )

        spec = specs["get_weather"]
        self.assertEqual(spec.allowed_properties, frozenset({"city", "days"}))
        self.assertEqual(spec.required_properties, frozenset({"city"}))
        self.assertEqual(spec.property_types, {"city": "string", "days": "integer"})


class MiniCPM5ParserTests(unittest.TestCase):
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

    def test_detect_and_parse_extracts_xml_tool_call(self) -> None:
        text = (
            "before <think>hidden reasoning</think>"
            " <function name=\"get_weather\">"
            "<param name=\"city\">Paris</param>"
            "<param name=\"days\">2</param>"
            "</function> after"
        )

        result = self.parser.detect_and_parse(text, self.tools)

        self.assertEqual(result.tool_calls, [ParsedToolCall("get_weather", {"city": "Paris", "days": 2})])
        self.assertNotIn("<think>", result.normal_text)
        self.assertNotIn("<function", result.normal_text)
        self.assertIn("before", result.normal_text)
        self.assertIn("after", result.normal_text)

    def test_detect_and_parse_keeps_unknown_tool_as_text(self) -> None:
        text = '<function name="unknown"><param name="city">Paris</param></function>'

        result = self.parser.detect_and_parse(text, self.tools)

        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.normal_text, text)

    def test_detect_and_parse_rejects_param_without_name(self) -> None:
        text = '<function name="get_weather"><param>Paris</param></function>'

        result = self.parser.detect_and_parse(text, self.tools)

        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.normal_text, text)

    def test_build_openai_tool_calls_serializes_arguments_as_json(self) -> None:
        calls = [ParsedToolCall("get_weather", {"city": "Paris", "days": 2})]

        tool_calls = build_openai_tool_calls("chatcmpl-123", 0, calls)

        self.assertEqual(tool_calls[0]["id"], "chatcmpl-123-call-0-0")
        self.assertEqual(tool_calls[0]["type"], "function")
        self.assertEqual(tool_calls[0]["index"], 0)
        self.assertEqual(tool_calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(tool_calls[0]["function"]["arguments"]), {"city": "Paris", "days": 2})
