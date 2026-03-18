import pathlib
import json

def parse_stats(path: pathlib.Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
        parsed = json.loads(raw)
        return {
            "max_input_length": parsed.inputLengths.p95,
            "max_label_length": parsed.labelLengths.p95

        }