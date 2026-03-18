#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from Seq2SeqCollator.constructor import Seq2SeqCollator
from Seq2SeqTransformer.constructor import Seq2SeqTransformer
from TokenizedJsonlDataset.constructor import TokenizedJsonlDataset
from args.parse import parse_args
from artifacts.save import save_artifacts
from device.build import build_device
from epoch.train import train_epoch
from loss.evaluate import evaluate_loss
from match.compute import compute_exact_match
from parameters.count import count_parameters
from seed.set import set_seed
from stats.parse import parse_stats
from vocab.read_size import read_vocab_size
from sequence.get_effective_lenght import get_effective_sequence_lengths


TOKENIZERS_ROOT = pathlib.Path("src/03_tokenizers")
DATASETS_ROOT = pathlib.Path("src/04_training_datasets")
PYTORCH_MODELS_ROOT = pathlib.Path("src/05_pytorch_models")

TOKENIZER_VOCAB_FILE = "tokenizer.vocab"
TRAINING_DATA_FILE = "train.jsonl"
VALIDATION_DATA_FILE = "validation.jsonl"
STATS_FILE = "stats.json"

# STATIC CONFIG
SEED = 7
LOG_FREQUENCY = 1
BOS_ID = 1
EOS_ID = 2
LABEL_PAD_ID = -100
GRAD_CLIP = 1.0
ACCUMULATION_STEPS = 16

# DYNAMIC CONFIG
CONFIG = {
    "exact_match_frequency": {"s": 1, "m": 2, "l": 3},
    "batch_size": {"s": 1, "m": 2, "l": 4},
    "effective_batch_size": {"s": 16, "m": 32, "l": 64},
    "epochs": {"s": 100, "m": 30, "l": 10},
    "early_stopping_patience": {"s": 20, "m": 8, "l": 3},
    "early_stopping_min_delta": {"s": 1e-5, "m": 1e-4, "l": 1e-3},
    "d_model": {"s": 128, "m": 256, "l": 512},
    "attention_heads": {"s": 4, "m": 4, "l": 8},
    "encoder_layers": {"s": 2, "m": 4, "l": 6},
    "decoder_layers": {"s": 2, "m": 4, "l": 6},
    "ff_dimension": {"s": 512, "m": 1024, "l": 2048},
    "dropout": {"s": 0.20, "m": 0.10, "l": 0.10},
    "learning_rate": {"s": 3e-4, "m": 2e-4, "l": 1e-4},
    "weight_decay": {"s": 5e-4, "m": 1e-4, "l": 1e-4},
}

def main() -> None:
    args = parse_args()
    language = args.language

    vocab_path = TOKENIZERS_ROOT / language / TOKENIZER_VOCAB_FILE
    training_dataset_path = DATASETS_ROOT / language / TRAINING_DATA_FILE
    validation_dataset_path = DATASETS_ROOT / language / VALIDATION_DATA_FILE
    dataset_stats_path = DATASETS_ROOT / language / STATS_FILE
    save_dir = PYTORCH_MODELS_ROOT / language

    vocab_size = read_vocab_size(vocab_path)
    pad_id = vocab_size

    train_dataset = TokenizedJsonlDataset(training_dataset_path, vocab_size)
    validation_dataset = TokenizedJsonlDataset(validation_dataset_path, vocab_size)

    stats = parse_stats(dataset_stats_path)
    dataset_scale = stats["dataset_scale"]
    dynamic_config = {
        key: values[dataset_scale]
        for key, values in CONFIG.items()
    }

    sequence_lengths = get_effective_sequence_lengths(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        max_input_length=stats["max_input_length"],
        max_label_length=stats["max_label_length"],
    )

    collator = Seq2SeqCollator(
        pad_id=pad_id,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        label_pad_id=LABEL_PAD_ID,
        max_input_length=sequence_lengths.max_input_length,
        max_label_length=sequence_lengths.max_label_length,
    )

    batch_size = dynamic_config["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    evaluation_train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    device = build_device(args.device)
    set_seed(SEED)

    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        pad_id=pad_id,
        d_model=dynamic_config["d_model"],
        num_heads=dynamic_config["attention_heads"],
        num_encoder_layers=dynamic_config["encoder_layers"],
        num_decoder_layers=dynamic_config["decoder_layers"],
        ffn_dim=dynamic_config["ff_dimension"],
        dropout=dynamic_config["dropout"],
        max_source_positions=sequence_lengths.max_source_positions,
        max_target_positions=sequence_lengths.max_target_positions,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=dynamic_config["learning_rate"],
        weight_decay=dynamic_config["weight_decay"],
    )

    max_generation_length = sequence_lengths.max_target_positions
    exact_match_frequency = dynamic_config["exact_match_frequency"]
    epochs = dynamic_config["epochs"]
    early_stopping_patience = dynamic_config["early_stopping_patience"]
    early_stopping_min_delta = dynamic_config["early_stopping_min_delta"]

    best_metrics: dict | None = None
    best_validation_loss: float | None = None
    epochs_without_improvement = 0

    print(f"Language: {language}")
    print(f"Dataset scale: {dataset_scale}")
    print(f"Train split: {training_dataset_path}")
    print(f"Validation split: {validation_dataset_path}")
    print(f"Tokenizer vocab: {vocab_path}")
    print(f"Vocab size: {vocab_size}")
    print(f"Pad id: {pad_id}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Effective batch size target: {dynamic_config['effective_batch_size']}")
    print(f"Batch size: {batch_size}")
    print(f"Accumulation steps (not yet wired): {ACCUMULATION_STEPS}")
    print(f"Effective max input length: {sequence_lengths.max_input_length}")
    print(f"Effective max label length: {sequence_lengths.max_label_length}")
    print(f"Max source positions: {sequence_lengths.max_source_positions}")
    print(f"Max target positions: {sequence_lengths.max_target_positions}")
    print(f"Parameter count: {count_parameters(model)}")
    print(f"Device: {device}")
    print(f"Save dir: {save_dir}")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            label_pad_id=LABEL_PAD_ID,
            grad_clip=GRAD_CLIP,
        )
        validation_loss = evaluate_loss(
            model,
            validation_loader,
            device=device,
            label_pad_id=LABEL_PAD_ID,
        )
        should_run_exact_match = (
            exact_match_frequency > 0
            and (epoch % exact_match_frequency == 0 or epoch == epochs)
        )

        train_exact_match: float | None = None
        validation_exact_match: float | None = None
        if should_run_exact_match:
            train_exact_match = compute_exact_match(
                model,
                evaluation_train_loader,
                device=device,
                bos_id=BOS_ID,
                eos_id=EOS_ID,
                max_generation_length=max_generation_length,
            )
            validation_exact_match = compute_exact_match(
                model,
                validation_loader,
                device=device,
                bos_id=BOS_ID,
                eos_id=EOS_ID,
                max_generation_length=max_generation_length,
            )

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_exact_match": train_exact_match,
            "validation_exact_match": validation_exact_match,
        }

        improved_validation = (
            best_validation_loss is None
            or validation_loss < best_validation_loss - early_stopping_min_delta
        )

        if improved_validation:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            best_metrics = metrics
            save_artifacts(
                save_dir=save_dir,
                metrics=metrics,
                model=model,
                optimizer=optimizer,
            )
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % LOG_FREQUENCY == 0 or epoch == epochs:
            log_message = (
                f"epoch={epoch} "
                f"train_loss={train_loss:.4f} "
                f"validation_loss={validation_loss:.4f}"
            )
            if train_exact_match is not None and validation_exact_match is not None:
                log_message = (
                    f"{log_message} "
                    f"train_exact_match={train_exact_match:.3f} "
                    f"validation_exact_match={validation_exact_match:.3f}"
                )
            print(log_message)

        if train_exact_match is not None and train_exact_match >= 1.0:
            print(f"Stopping early at epoch {epoch}: train_exact_match reached 1.000")
            break

        if epochs_without_improvement >= early_stopping_patience:
            print(
                "Stopping early at epoch "
                f"{epoch}: validation loss did not improve for "
                f"{early_stopping_patience} epochs"
            )
            break

    if best_metrics is not None:
        print("Best train exact match snapshot:")
        print(json.dumps(best_metrics, indent=2))


if __name__ == "__main__":
    main()
