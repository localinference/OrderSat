from __future__ import annotations

import json
import pathlib
import tempfile
from dataclasses import dataclass


@dataclass(frozen=True)
class PreparedCorpus:
    source_path: pathlib.Path
    prepared_path: pathlib.Path
    sample_count: int
    text_count: int


def _stable_json_value(value):
    if isinstance(value, list):
        return [_stable_json_value(item) for item in value]
    if isinstance(value, dict):
        normalized = {}
        for key in sorted(value):
            normalized[key] = _stable_json_value(value[key])
        return normalized
    return value


def _stable_stringify(value) -> str:
    return json.dumps(
        _stable_json_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _parse_corpus_line(line: str, line_number: int) -> tuple[str, object]:
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSONL at line {line_number}: {error.msg}"
        ) from error

    if not isinstance(parsed, dict):
        raise ValueError(f"Expected object at line {line_number}")
    input_text = parsed.get("input")
    if not isinstance(input_text, str):
        raise ValueError(f"Expected string input at line {line_number}")
    if "output" not in parsed:
        raise ValueError(f"Missing output at line {line_number}")
    return input_text, parsed["output"]


def prepare_corpus(source_path: pathlib.Path, workdir: pathlib.Path) -> PreparedCorpus:
    raw = source_path.read_text(encoding="utf8")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf8",
        newline="\n",
        suffix=".txt",
        prefix="tokenizer-training-",
        dir=workdir,
        delete=False,
    ) as handle:
        prepared_path = pathlib.Path(handle.name)
        sample_count = 0
        text_count = 0

        for line_number, line in enumerate(lines, start=1):
            input_text, output_value = _parse_corpus_line(line, line_number)
            output_text = _stable_stringify(output_value)

            if input_text:
                handle.write(input_text)
                handle.write("\n")
                text_count += 1

            if output_text:
                handle.write(output_text)
                handle.write("\n")
                text_count += 1

            sample_count += 1

    return PreparedCorpus(
        source_path=source_path,
        prepared_path=prepared_path,
        sample_count=sample_count,
        text_count=text_count,
    )
