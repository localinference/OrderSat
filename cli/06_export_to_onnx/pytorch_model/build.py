from __future__ import annotations

import importlib.util
import pathlib

import torch

from checkpoint.load import ExportModelConfig

SEQ2SEQ_CONSTRUCTOR_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "05_train_pytorch_models"
    / "Seq2SeqTransformer"
    / "constructor.py"
)


def build_pytorch_model(
    *,
    checkpoint: dict,
    model_config: ExportModelConfig,
) -> torch.nn.Module:
    seq2seq_transformer_class = load_seq2seq_transformer_class()
    model = seq2seq_transformer_class(
        vocab_size=model_config.vocab_size,
        pad_id=model_config.pad_id,
        d_model=model_config.d_model,
        num_heads=model_config.attention_heads,
        num_encoder_layers=model_config.encoder_layers,
        num_decoder_layers=model_config.decoder_layers,
        ffn_dim=model_config.ff_dimension,
        dropout=model_config.dropout,
        max_source_positions=model_config.max_source_positions,
        max_target_positions=model_config.max_target_positions,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def load_seq2seq_transformer_class() -> type[torch.nn.Module]:
    if not SEQ2SEQ_CONSTRUCTOR_PATH.exists():
        raise SystemExit(
            "Seq2SeqTransformer constructor does not exist: "
            f"{SEQ2SEQ_CONSTRUCTOR_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "seq2seq_transformer_constructor",
        SEQ2SEQ_CONSTRUCTOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise SystemExit(
            "Failed to load Seq2SeqTransformer constructor module: "
            f"{SEQ2SEQ_CONSTRUCTOR_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    seq2seq_transformer_class = getattr(module, "Seq2SeqTransformer", None)
    if seq2seq_transformer_class is None:
        raise SystemExit(
            "Seq2SeqTransformer class is missing from constructor module: "
            f"{SEQ2SEQ_CONSTRUCTOR_PATH}"
        )

    return seq2seq_transformer_class
