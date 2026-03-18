from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch.utils.data import Sampler

from TokenizedJsonlDataset.constructor import TokenizedJsonlDataset


@dataclass(frozen=True)
class BatchPlanSummary:
    batch_count: int
    sample_count: int
    average_batch_size: float
    max_batch_size_observed: int
    average_batch_tokens: float
    max_batch_tokens_observed: int
    target_tokens_per_batch: int
    max_batch_size: int
    shuffle: bool

    def to_dict(self) -> dict:
        return asdict(self)


class TokenBudgetBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: TokenizedJsonlDataset,
        *,
        target_tokens_per_batch: int,
        max_batch_size: int,
        shuffle: bool,
        seed: int = 0,
        sort_pool_size: int | None = None,
    ) -> None:
        if target_tokens_per_batch < 1:
            raise SystemExit("target_tokens_per_batch must be at least 1")
        if max_batch_size < 1:
            raise SystemExit("max_batch_size must be at least 1")

        self.dataset = dataset
        self.target_tokens_per_batch = target_tokens_per_batch
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.sort_pool_size = sort_pool_size or max(max_batch_size * 16, 64)
        self._iteration_index = 0
        self._cached_iteration_index: int | None = None
        self._cached_batches: list[list[int]] | None = None
        self._cached_summary: BatchPlanSummary | None = None

    def describe_current_plan(self) -> BatchPlanSummary:
        _, summary = self._get_plan(self._iteration_index)
        return summary

    def __len__(self) -> int:
        batches, _ = self._get_plan(self._iteration_index)
        return len(batches)

    def __iter__(self):
        batches, _ = self._get_plan(self._iteration_index)
        self._iteration_index += 1
        for batch in batches:
            yield batch

    def _get_plan(self, iteration_index: int) -> tuple[list[list[int]], BatchPlanSummary]:
        if self._cached_iteration_index == iteration_index:
            assert self._cached_batches is not None
            assert self._cached_summary is not None
            return self._cached_batches, self._cached_summary

        batches, summary = self._build_plan(iteration_index)
        self._cached_iteration_index = iteration_index
        self._cached_batches = batches
        self._cached_summary = summary
        return batches, summary

    def _build_plan(self, iteration_index: int) -> tuple[list[list[int]], BatchPlanSummary]:
        ordered_indices = self._build_index_order(iteration_index)
        batches: list[list[int]] = []
        batch_token_costs: list[int] = []

        current_batch: list[int] = []
        current_max_input = 0
        current_max_target = 0

        for index in ordered_indices:
            input_length = self.dataset.get_input_length(index)
            target_length = self.dataset.get_target_length(index)

            if not current_batch:
                current_batch = [index]
                current_max_input = input_length
                current_max_target = target_length
                continue

            projected_max_input = max(current_max_input, input_length)
            projected_max_target = max(current_max_target, target_length)
            projected_batch_size = len(current_batch) + 1
            projected_batch_tokens = projected_batch_size * (
                projected_max_input + projected_max_target
            )

            should_close_batch = (
                len(current_batch) >= self.max_batch_size
                or projected_batch_tokens > self.target_tokens_per_batch
            )
            if should_close_batch:
                batches.append(list(current_batch))
                batch_token_costs.append(
                    len(current_batch) * (current_max_input + current_max_target)
                )
                current_batch = [index]
                current_max_input = input_length
                current_max_target = target_length
                continue

            current_batch.append(index)
            current_max_input = projected_max_input
            current_max_target = projected_max_target

        if current_batch:
            batches.append(list(current_batch))
            batch_token_costs.append(
                len(current_batch) * (current_max_input + current_max_target)
            )

        if self.shuffle and len(batches) > 1:
            generator = torch.Generator()
            generator.manual_seed(self.seed + iteration_index + 1_000_000)
            permutation = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[position] for position in permutation]
            batch_token_costs = [batch_token_costs[position] for position in permutation]

        sample_count = len(ordered_indices)
        batch_count = len(batches)
        average_batch_size = 0.0
        average_batch_tokens = 0.0
        max_batch_size_observed = 0
        max_batch_tokens_observed = 0
        if batch_count > 0:
            average_batch_size = sample_count / batch_count
            average_batch_tokens = sum(batch_token_costs) / batch_count
            max_batch_size_observed = max(len(batch) for batch in batches)
            max_batch_tokens_observed = max(batch_token_costs)

        summary = BatchPlanSummary(
            batch_count=batch_count,
            sample_count=sample_count,
            average_batch_size=average_batch_size,
            max_batch_size_observed=max_batch_size_observed,
            average_batch_tokens=average_batch_tokens,
            max_batch_tokens_observed=max_batch_tokens_observed,
            target_tokens_per_batch=self.target_tokens_per_batch,
            max_batch_size=self.max_batch_size,
            shuffle=self.shuffle,
        )
        return batches, summary

    def _build_index_order(self, iteration_index: int) -> list[int]:
        indices = list(range(len(self.dataset)))
        if not indices:
            return []

        if not self.shuffle:
            return sorted(indices, key=self._sort_key)

        generator = torch.Generator()
        generator.manual_seed(self.seed + iteration_index)
        shuffled_positions = torch.randperm(len(indices), generator=generator).tolist()
        shuffled_indices = [indices[position] for position in shuffled_positions]

        ordered_indices: list[int] = []
        for start in range(0, len(shuffled_indices), self.sort_pool_size):
            pool = shuffled_indices[start : start + self.sort_pool_size]
            pool.sort(key=self._sort_key)
            ordered_indices.extend(pool)
        return ordered_indices

    def _sort_key(self, index: int) -> tuple[int, int, int]:
        return (
            self.dataset.get_sequence_token_count(index),
            self.dataset.get_target_length(index),
            self.dataset.get_input_length(index),
        )
