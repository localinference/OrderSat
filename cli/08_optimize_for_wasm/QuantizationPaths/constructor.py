import pathlib
from dataclasses import dataclass

@dataclass(frozen=True)
class QuantizationPaths:
    source_model_path: pathlib.Path
    source_config_path: pathlib.Path
    source_metrics_path: pathlib.Path
    source_tokenizer_model_path: pathlib.Path
    source_tokenizer_vocab_path: pathlib.Path
    quantized_dir: pathlib.Path
    quantized_model_path: pathlib.Path
    quantized_config_path: pathlib.Path
    quantized_metrics_path: pathlib.Path
    quantized_tokenizer_model_path: pathlib.Path
    quantized_tokenizer_vocab_path: pathlib.Path