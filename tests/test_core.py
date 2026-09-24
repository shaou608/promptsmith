"""Fast regression tests for PromptSmith's notebook-only helpers.

These tests load selected definitions directly from the notebook, so they do not
download models or require a GPU.
"""

from __future__ import annotations

import ast
import builtins
import json
import re
from pathlib import Path
from typing import Any, Dict


NOTEBOOK = Path(__file__).parents[1] / "COCONUT.ipynb"
TARGETS = {
    "extract_first_json_block",
    "safe_json_loads",
    "_empty_parts",
    "_norm",
    "sanitize_parts",
    "merge_with_lock",
    "infer_schema_mode_b",
    "enrich_constraints_if_needed",
    "_concat_unique",
    "apply_additions",
    "generate_intent_variants",
    "ask_int",
    "ask_float",
    "ask_size",
}
CONSTANTS = {
    "KEY_FIELDS",
    "SCHEMA_KEYS",
    "NONE_SYNONYMS",
    "GENERIC_TAGS",
    "SYSTEM_STRICT_JSON",
}


def load_helpers() -> dict[str, Any]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    selected: list[ast.stmt] = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "\n".join(
            line for line in "".join(cell.get("source", [])).splitlines()
            if not line.lstrip().startswith(("!", "%"))
        )
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in TARGETS:
                selected.append(node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                names = {
                    target.id
                    for target in (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                    if isinstance(target, ast.Name)
                }
                if names & CONSTANTS:
                    selected.append(node)

    namespace: dict[str, Any] = {
        "Any": Any,
        "Dict": Dict,
        "json": json,
        "re": re,
    }
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(NOTEBOOK), "exec"), namespace)
    return namespace


def with_input(value: str, function, *args):
    original = builtins.input
    builtins.input = lambda _prompt="": value
    try:
        return function(*args)
    finally:
        builtins.input = original


def main() -> None:
    helpers = load_helpers()

    parsed = helpers["safe_json_loads"]('{"text":"a } symbol","nested":{"ok":true}}')
    assert parsed == {"text": "a } symbol", "nested": {"ok": True}}

    base = {
        "subject_scene": "a cat",
        "colors": "blue",
        "foreground_objects": "",
        "background_mood": "calm",
        "style": "cartoon",
        "camera": "close-up",
        "lighting": "soft",
        "constraints": "",
    }
    proposed = {**base, "style": "photorealistic", "camera": "wide shot"}
    merged = helpers["merge_with_lock"](base, proposed, {"style", "camera"})
    assert merged["style"] == "cartoon"
    assert merged["camera"] == "close-up"

    helpers["trim_to_tokens"] = lambda text, *_args, **_kwargs: text
    helpers["join_prompt"] = lambda parts: json.dumps(parts, sort_keys=True)
    helpers["llm_chat"] = lambda *_args, **_kwargs: "not JSON"
    inferred = helpers["infer_schema_mode_b"]("a cat", "a cat")
    assert inferred["subject_scene"] == "a cat"
    assert helpers["enrich_constraints_if_needed"]("a cat", base, set()) == base
    fallback_variants = helpers["generate_intent_variants"]("a cat", base, {"style"}, 3)
    assert len(fallback_variants) == 3
    assert all(v["parts"]["style"] == "cartoon" for v in fallback_variants)

    helpers["llm_chat"] = lambda *_args, **_kwargs: json.dumps({
        "variants": [{
            "intent_tag": "Contradiction test",
            "why": "Checks locked fields.",
            "additions": {"style": "photorealistic", "lighting": "rim light"},
        }]
    })
    locked_variants = helpers["generate_intent_variants"]("a cat", base, {"style"}, 1)
    assert locked_variants[0]["parts"]["style"] == "cartoon"
    assert locked_variants[0]["parts"]["lighting"] == "soft, rim light"

    assert with_input("not-a-number", helpers["ask_int"], "", 30, 1) == 30
    assert with_input("-5", helpers["ask_int"], "", 30, 1) == 1
    assert with_input("bad", helpers["ask_float"], "", 6.0, 0.0) == 6.0
    assert with_input("513x512", helpers["ask_size"], "") == (1024, 1024)
    assert with_input("512x768", helpers["ask_size"], "") == (512, 768)

    print("PromptSmith core regression tests passed.")


if __name__ == "__main__":
    main()
