from __future__ import annotations

import ast
import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

_FUNC_NAME_V1_REGEX = re.compile(r"<function\s+name=[\'\"]([^\'\"]+)[\'\"][^>]*>")
_PARAM_WITH_NAME_REGEX = re.compile(
    r"<param\s+name=[\'\"]([^\'\"]+)[\'\"]>([\s\S]*?)</param>", re.DOTALL
)
_PARAM_MISSING_NAME_REGEX = re.compile(r"<param(?![^>]*\bname=)[^>]*>", re.DOTALL)
_THINK_BLOCK_REGEX = re.compile(r"<think>[\s\S]*?</think>", re.DOTALL)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    allowed_properties: frozenset[str]
    required_properties: frozenset[str]
    property_types: dict[str, str]


@dataclass(frozen=True)
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ParseResult:
    normal_text: str
    tool_calls: list[ParsedToolCall]


def _unwrap(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _to_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "dict"):
        return dict(value.dict())
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def build_tool_specs(tools: Sequence[Any] | None) -> dict[str, ToolSpec]:
    specs: dict[str, ToolSpec] = {}
    for tool in tools or []:
        tool_obj = _to_mapping(tool)
        fn = _unwrap(tool_obj, "function", {})
        fn_obj = _to_mapping(fn)
        name = str(_unwrap(fn_obj, "name", "") or "").strip()
        if not name:
            continue
        params = _unwrap(fn_obj, "parameters", {})
        params_obj = _to_mapping(params) if isinstance(params, Mapping) or hasattr(params, "__dict__") else {}
        props = params_obj.get("properties", {}) if isinstance(params_obj, dict) else {}
        if not isinstance(props, dict):
            props = {}
        required = params_obj.get("required", []) if isinstance(params_obj, dict) else []
        if not isinstance(required, (list, tuple, set)):
            required = []
        property_types: dict[str, str] = {}
        for prop_name, schema in props.items():
            schema_obj = _to_mapping(schema)
            prop_type = schema_obj.get("type")
            if isinstance(prop_type, str):
                property_types[str(prop_name)] = prop_type
        specs[name] = ToolSpec(
            name=name,
            allowed_properties=frozenset(str(k) for k in props.keys()),
            required_properties=frozenset(str(x) for x in required),
            property_types=property_types,
        )
    return specs


def parse_arguments(json_value: str | None) -> tuple[Any, bool]:
    try:
        try:
            parsed_value = json.loads(json_value)  # type: ignore[arg-type]
        except (json.JSONDecodeError, TypeError):
            parsed_value = ast.literal_eval(json_value)  # type: ignore[arg-type]
        return parsed_value, True
    except (ValueError, SyntaxError, TypeError):
        return json_value, False


def _strip_think_blocks(text: str) -> str:
    return _THINK_BLOCK_REGEX.sub("", text)


class MiniCPM5Parser:
    """Minimal port of SGLang's MiniCPM5 XML tool-call detector."""

    bot_token = "<function"
    eot_token = "</function>"
    func_call_regex = r"<function.*?</function>"

    def __init__(self) -> None:
        self._regex = re.compile(self.func_call_regex, re.DOTALL)

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: Sequence[Any] | None) -> ParseResult:
        if self.bot_token not in text:
            return ParseResult(normal_text=_strip_think_blocks(text), tool_calls=[])

        specs = build_tool_specs(tools)
        tool_names = set(specs)
        normal_parts: list[str] = []
        calls: list[ParsedToolCall] = []

        last_end = 0
        for match in self._regex.finditer(text):
            if match.start() > last_end:
                normal_parts.append(text[last_end : match.start()])

            block = match.group(0)
            parsed = self._parse_block(block, specs, tool_names)
            if parsed is None:
                normal_parts.append(block)
            else:
                calls.append(parsed)

            last_end = match.end()

        if last_end < len(text):
            normal_parts.append(text[last_end:])

        return ParseResult(normal_text=_strip_think_blocks("".join(normal_parts)), tool_calls=calls)

    def _parse_block(
        self,
        block: str,
        specs: dict[str, ToolSpec],
        tool_names: set[str],
    ) -> ParsedToolCall | None:
        func_name = None
        arguments: dict[str, Any] = {}
        parsed_ok = False
        param_invalid = False

        try:
            root = ET.fromstring(block)
            func_node = root if root.tag == "function" else root.find("function")
            if func_node is not None:
                func_name = (func_node.attrib.get("name") or "").strip()

            args_node = func_node.find("arguments") if func_node is not None else None
            param_nodes = []
            if func_node is not None:
                param_nodes = list(func_node.findall("param"))
                if args_node is not None and not param_nodes:
                    param_nodes = list(args_node.findall("param"))

            if func_node is not None:
                seen_keys: set[str] = set()
                allowed_props = specs.get(func_name, ToolSpec("", frozenset(), frozenset(), {})).allowed_properties if func_name in specs else frozenset()
                for param in param_nodes:
                    key = (param.attrib.get("name") or "").strip()
                    if not key:
                        param_invalid = True
                        break
                    if allowed_props and key not in allowed_props:
                        param_invalid = True
                        break
                    if key in seen_keys:
                        param_invalid = True
                        break
                    seen_keys.add(key)
                    val_text = (param.text or "").strip()
                    arg_type = specs.get(func_name, ToolSpec("", frozenset(), frozenset(), {})).property_types.get(key) if func_name in specs else None
                    if arg_type != "string":
                        parsed_val, _ = parse_arguments(val_text)
                        arguments[key] = parsed_val
                    else:
                        arguments[key] = val_text

            parsed_ok = bool(func_name)
        except Exception:
            parsed_ok = False

        if not parsed_ok:
            try:
                m_fn = _FUNC_NAME_V1_REGEX.search(block)
                if m_fn:
                    func_name = (m_fn.group(1) or "").strip()
                if not func_name:
                    return None
                if func_name not in tool_names:
                    return None
                has_invalid_param = _PARAM_MISSING_NAME_REGEX.search(block) is not None
                seen_keys = set()
                spec = specs.get(func_name)
                allowed_props = spec.allowed_properties if spec else frozenset()
                for pm in _PARAM_WITH_NAME_REGEX.finditer(block):
                    key = (pm.group(1) or "").strip()
                    if not key:
                        has_invalid_param = True
                        break
                    if allowed_props and key not in allowed_props:
                        has_invalid_param = True
                        break
                    if key in seen_keys:
                        has_invalid_param = True
                        break
                    seen_keys.add(key)
                    val_text = pm.group(2) or ""
                    if val_text.startswith("<![CDATA[") and val_text.endswith("]]>"):
                        val_text = val_text[len("<![CDATA[") : -len("]]>")]
                    val_text = val_text.strip()
                    arg_type = spec.property_types.get(key) if spec else None
                    if arg_type != "string":
                        parsed_val, _ = parse_arguments(val_text)
                        arguments[key] = parsed_val
                    else:
                        arguments[key] = val_text
                if has_invalid_param:
                    return None
                parsed_ok = True
            except Exception:
                return None

        if not func_name or func_name not in tool_names or param_invalid:
            return None

        spec = specs.get(func_name)
        required = spec.required_properties if spec else frozenset()
        if required and not required.issubset(arguments.keys()):
            return None

        if not parsed_ok:
            return None

        return ParsedToolCall(name=func_name, arguments=arguments)


def build_openai_tool_calls(
    completion_id: str,
    choice_index: int,
    calls: Sequence[ParsedToolCall],
    start_index: int = 0,
) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for offset, call in enumerate(calls, start=start_index):
        tool_calls.append(
            {
                "id": f"{completion_id}-call-{choice_index}-{offset}",
                "type": "function",
                "index": offset,
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
        )
    return tool_calls
