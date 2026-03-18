import pathlib
import json

def parse_stats(path: pathlib.Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
        parsed = json.loads(raw)

        sample_count = parsed["sampleCount"]
        if sample_count < 10_000:
            sample_scale = "s"
        elif sample_count < 100_000:
            sample_scale = "m"
        else:
            sample_scale = "l"

        return {
            "sample_scale": sample_scale,
            "max_input_length": parsed["inputLengths"]["p95"],
            "max_label_length": parsed["labelLengths"]["p95"]
        }