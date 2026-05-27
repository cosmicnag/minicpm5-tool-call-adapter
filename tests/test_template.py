from __future__ import annotations

import json
import unittest
from pathlib import Path

from jinja2 import Environment


class MiniCPM5TemplateTests(unittest.TestCase):
    def test_system_prompt_appends_tools_once(self) -> None:
        template_path = (
            Path(__file__).resolve().parents[2]
            / "chat-templates"
            / "minicpm5"
            / "chat_template.jinja"
        )
        template_text = template_path.read_text(encoding="utf-8")

        env = Environment()
        env.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii=kwargs.get("ensure_ascii", False))
        template = env.from_string(template_text)

        rendered = template.render(
            bos_token="<s>",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that can call tools."},
                {"role": "user", "content": "hello"},
            ],
            tools=[
                {
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                }
            ],
            add_generation_prompt=True,
            enable_thinking=True,
        )

        self.assertIn("You are a helpful assistant that can call tools.", rendered)
        self.assertEqual(rendered.count("# Tools"), 1)
        self.assertEqual(rendered.count("Tool usage guidelines"), 1)


if __name__ == "__main__":
    unittest.main()
