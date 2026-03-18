import pathlib
import json

def parse_stats(path: pathlib.Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
        parsed = json.loads(raw)

        sample_count = parsed["sampleCount"]
        if sample_count < 10_000:
            dataset_scale = "s"
        elif sample_count < 100_000:
            dataset_scale = "m"
        else:
            dataset_scale = "l"

        return {
            "dataset_scale": dataset_scale,
            "max_input_length": parsed["inputLengths"]["max"],
            "max_label_length": parsed["labelLengths"]["max"]
        }