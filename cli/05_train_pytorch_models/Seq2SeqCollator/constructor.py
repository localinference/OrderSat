import torch

class Seq2SeqCollator:
    def __init__(
        self,
        *,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        label_pad_id: int,
        max_input_length: int | None = None,
        max_label_length: int | None = None,
    ) -> None:
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.label_pad_id = label_pad_id
        self.max_input_length = max_input_length
        self.max_label_length = max_label_length

    def __call__(self, items: list[dict]) -> dict:
        if torch is None:
            raise SystemExit(
                "PyTorch is required to collate batches. Install torch first."
            )

        truncated_inputs = [
            self._truncate_input(item["input_ids"]) for item in items
        ]
        truncated_labels = [
            self._truncate_labels(item["labels"]) for item in items
        ]

        decoder_inputs = [[self.bos_id, *labels] for labels in truncated_labels]
        decoder_targets = [[*labels, self.eos_id] for labels in truncated_labels]

        input_ids = self._pad_tokens(truncated_inputs, pad_value=self.pad_id)
        attention_mask = input_ids.ne(self.pad_id).to(dtype=torch.int32)
        decoder_input_ids = self._pad_tokens(
            decoder_inputs,
            pad_value=self.pad_id,
        )
        decoder_attention_mask = decoder_input_ids.ne(self.pad_id).to(
            dtype=torch.int32
        )
        labels = self._pad_tokens(
            decoder_targets,
            pad_value=self.label_pad_id,
        )

        return {
            "sample_ids": [item["sample_id"] for item in items],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
            "target_token_ids": [list(tokens) for tokens in truncated_labels],
            "input_lengths": torch.tensor(
                [len(tokens) for tokens in truncated_inputs],
                dtype=torch.int32,
            ),
            "label_lengths": torch.tensor(
                [len(tokens) for tokens in decoder_targets],
                dtype=torch.int32,
            ),
        }

    def _truncate_input(self, token_ids: list[int]) -> list[int]:
        if self.max_input_length is None:
            return list(token_ids)
        return list(token_ids[: self.max_input_length])

    def _truncate_labels(self, token_ids: list[int]) -> list[int]:
        if self.max_label_length is None:
            return list(token_ids)
        if self.max_label_length < 1:
            raise SystemExit("--max-label-length must be at least 1")
        return list(token_ids[: self.max_label_length])

    @staticmethod
    def _pad_tokens(sequences: list[list[int]], pad_value: int) -> torch.Tensor:
        max_length = max(len(sequence) for sequence in sequences)
        padded = [
            sequence + [pad_value] * (max_length - len(sequence))
            for sequence in sequences
        ]
        return torch.tensor(padded, dtype=torch.int32)
