import TokenizedJsonlDataset
import EffectiveSequenceLengths

def get_effective_sequence_lengths(
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