#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None
    F = None
    nn = None
    DataLoader = None
    Dataset = object

    from Seq2SeqCollator.constructor import Seq2SeqCollator

    from TokenizedJsonlDataset.constructor import TokenizedJsonlDataset

    from TinySeq2SeqTransformer.constructor import TinySeq2SeqTransformer

    from args.parse import parse_args

    from vocab.read_size import read_vocab_size


TOKENIZERS_ROOT = pathlib.Path("src/03_tokenizers")
DATASETS_ROOT = pathlib.Path("src/04_training_datasets")
PYTORCH_MODELS_ROOT = pathlib.Path("src/05_pytorch_models")

DATASET_STATS_FILE = "stats.json"
TRAINING_DATA_FILE = "train.jsonl"
VALIDATION_DATA_FILE = "validation.jsonl"
TOKENIZER_VOCAB_FILE = "tokenizer.vocab"

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





@dataclass(frozen=True)
class SplitPaths:
    train_path: pathlib.Path
    validation_path: pathlib.Path
    vocab_path: pathlib.Path


@dataclass(frozen=True)
class EffectiveSequenceLengths:
    max_input_length: int
    max_label_length: int
    max_source_positions: int
    max_target_positions: int







def build_device(device_name: str) -> str:
    if device_name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_name





def resolve_effective_sequence_lengths(
    *,
    train_dataset: TokenizedJsonlDataset,
    validation_dataset: TokenizedJsonlDataset,
    max_input_length: int,
    max_label_length: int,
) -> EffectiveSequenceLengths:
    observed_max_input_length = max(
        max(len(record["input_ids"]) for record in train_dataset.records),
        max(len(record["input_ids"]) for record in validation_dataset.records),
    )
    observed_max_label_length = max(
        max(len(record["labels"]) for record in train_dataset.records),
        max(len(record["labels"]) for record in validation_dataset.records),
    )

    effective_input_length = min(observed_max_input_length, max_input_length)
    effective_label_length = min(observed_max_label_length, max_label_length)

    return EffectiveSequenceLengths(
        max_input_length=effective_input_length,
        max_label_length=effective_label_length,
        max_source_positions=effective_input_length,
        max_target_positions=effective_label_length + 1,
    )


def count_parameters(model: TinySeq2SeqTransformer) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def greedy_generate(
    model: TinySeq2SeqTransformer,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_generation_length: int,
) -> list[list[int]]:
    model.eval()
    memory, source_padding_mask = model.encode(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    batch_size = input_ids.size(0)
    generated = torch.full(
        (batch_size, 1),
        fill_value=bos_id,
        dtype=torch.long,
        device=input_ids.device,
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

    for _ in range(max_generation_length):
        logits = model.decode_step(
            decoder_input_ids=generated,
            memory=memory,
            source_padding_mask=source_padding_mask,
        )
        next_token = logits[:, -1, :].argmax(dim=-1)
        next_token = torch.where(
            finished,
            torch.full_like(next_token, eos_id),
            next_token,
        )
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        finished = finished | next_token.eq(eos_id)
        if bool(finished.all()):
            break

    outputs: list[list[int]] = []
    for sequence in generated[:, 1:].tolist():
        trimmed: list[int] = []
        for token_id in sequence:
            if token_id == eos_id:
                break
            trimmed.append(token_id)
        outputs.append(trimmed)

    return outputs


def compute_exact_match(
    model: TinySeq2SeqTransformer,
    loader: DataLoader,
    *,
    device: str,
    bos_id: int,
    eos_id: int,
    max_generation_length: int,
) -> float:
    matches = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            predictions = greedy_generate(
                model,
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                bos_id=bos_id,
                eos_id=eos_id,
                max_generation_length=max_generation_length,
            )
            targets = batch["target_token_ids"]
            for predicted, target in zip(predictions, targets):
                total += 1
                if predicted == target:
                    matches += 1

    if total == 0:
        return 0.0

    return matches / total


def compute_loss(
    model: TinySeq2SeqTransformer,
    batch: dict,
    *,
    device: str,
    label_pad_id: int,
) -> torch.Tensor:
    logits = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        decoder_input_ids=batch["decoder_input_ids"].to(device),
    )
    labels = batch["labels"].to(device)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=label_pad_id,
    )


def evaluate_loss(
    model: TinySeq2SeqTransformer,
    loader: DataLoader,
    *,
    device: str,
    label_pad_id: int,
) -> float:
    model.eval()
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in loader:
            loss = compute_loss(
                model,
                batch,
                device=device,
                label_pad_id=label_pad_id,
            )
            total_loss += float(loss.item())
            batch_count += 1

    if batch_count == 0:
        return 0.0

    return total_loss / batch_count


def train_epoch(
    model: TinySeq2SeqTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: str,
    label_pad_id: int,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    batch_count = 0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(
            model,
            batch,
            device=device,
            label_pad_id=label_pad_id,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item())
        batch_count += 1

    return total_loss / batch_count


def save_artifacts(
    *,
    save_dir: pathlib.Path,
    metrics: dict,
    model: TinySeq2SeqTransformer,
    optimizer: torch.optim.Optimizer,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.json"
    checkpoint_path = save_dir / "best.pt"

    metrics_path.write_text(f"{json.dumps(metrics, indent=2)}\n", encoding="utf8")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )


def main() -> None:
    args = parse_args()


    split_paths = resolve_paths(args)
    save_dir = resolve_save_dir(args)
    vocab_size = read_vocab_size(split_paths.vocab_path)
    pad_id = vocab_size

    train_dataset = TokenizedJsonlDataset(split_paths.train_path, vocab_size=vocab_size)
    validation_dataset = TokenizedJsonlDataset(
        split_paths.validation_path,
        vocab_size=vocab_size,
    )

    sequence_lengths = resolve_effective_sequence_lengths(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        max_input_length=args.max_input_length,
        max_label_length=args.max_label_length,
    )

    collator = Seq2SeqCollator(
        pad_id=pad_id,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        label_pad_id=args.label_pad_id,
        max_input_length=sequence_lengths.max_input_length,
        max_label_length=sequence_lengths.max_label_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    evaluation_train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    device = build_device(args.device)
    set_seed(args.seed)

    model = TinySeq2SeqTransformer(
        vocab_size=vocab_size,
        pad_id=pad_id,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        max_source_positions=sequence_lengths.max_source_positions,
        max_target_positions=sequence_lengths.max_target_positions,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if args.max_generation_length is None:
        max_generation_length = sequence_lengths.max_target_positions
    else:
        max_generation_length = args.max_generation_length

    best_metrics: dict | None = None
    best_validation_loss: float | None = None
    epochs_without_improvement = 0

    print(f"Language: {args.language}")
    print(f"Train split: {split_paths.train_path}")
    print(f"Validation split: {split_paths.validation_path}")
    print(f"Tokenizer vocab: {split_paths.vocab_path}")
    print(f"Vocab size: {vocab_size}")
    print(f"Pad id: {pad_id}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Observed max input length: {max(len(record['input_ids']) for record in train_dataset.records + validation_dataset.records)}")
    print(f"Observed max label length: {max(len(record['labels']) for record in train_dataset.records + validation_dataset.records)}")
    print(f"Effective max input length: {sequence_lengths.max_input_length}")
    print(f"Effective max label length: {sequence_lengths.max_label_length}")
    print(f"Max source positions: {sequence_lengths.max_source_positions}")
    print(f"Max target positions: {sequence_lengths.max_target_positions}")
    print(f"Parameter count: {count_parameters(model)}")
    print(f"Device: {device}")
    print(f"Save dir: {save_dir}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            label_pad_id=args.label_pad_id,
            grad_clip=args.grad_clip,
        )
        validation_loss = evaluate_loss(
            model,
            validation_loader,
            device=device,
            label_pad_id=args.label_pad_id,
        )
        should_run_exact_match = (
            args.exact_match_every > 0
            and (
                epoch % args.exact_match_every == 0
                or epoch == args.epochs
            )
        )
        train_exact_match: float | None = None
        validation_exact_match: float | None = None
        if should_run_exact_match:
            train_exact_match = compute_exact_match(
                model,
                evaluation_train_loader,
                device=device,
                bos_id=args.bos_id,
                eos_id=args.eos_id,
                max_generation_length=max_generation_length,
            )
            validation_exact_match = compute_exact_match(
                model,
                validation_loader,
                device=device,
                bos_id=args.bos_id,
                eos_id=args.eos_id,
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
            or validation_loss < best_validation_loss - args.early_stopping_min_delta
        )

        if improved_validation:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if best_metrics is None:
            best_metrics = metrics
            save_artifacts(
                save_dir=save_dir,
                metrics=metrics,
                model=model,
                optimizer=optimizer,
            )
        elif improved_validation:
            best_metrics = metrics
            save_artifacts(
                save_dir=save_dir,
                metrics=metrics,
                model=model,
                optimizer=optimizer,
            )

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            log_message = (
                "epoch="
                f"{epoch} "
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

        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                "Stopping early at epoch "
                f"{epoch}: validation loss did not improve for "
                f"{args.early_stopping_patience} epochs"
            )
            break

    if best_metrics is not None:
        print("Best train exact match snapshot:")
        print(json.dumps(best_metrics, indent=2))


if __name__ == "__main__":
    main()
