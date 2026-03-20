from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckpointScore:
    epoch: int
    validation_loss: float
    validation_exact_match: float | None


def build_checkpoint_score(*, epoch: int, validation_loss: float, validation_exact_match: float | None) -> CheckpointScore:
    return CheckpointScore(
        epoch=epoch,
        validation_loss=validation_loss,
        validation_exact_match=validation_exact_match,
    )


def is_better_checkpoint(
    candidate: CheckpointScore,
    incumbent: CheckpointScore | None,
) -> bool:
    if incumbent is None:
        return True

    candidate_exact_match = candidate.validation_exact_match
    incumbent_exact_match = incumbent.validation_exact_match

    if candidate_exact_match is not None and incumbent_exact_match is not None:
        if candidate_exact_match != incumbent_exact_match:
            return candidate_exact_match > incumbent_exact_match
        if candidate.validation_loss != incumbent.validation_loss:
            return candidate.validation_loss < incumbent.validation_loss
        return candidate.epoch < incumbent.epoch

    if candidate_exact_match is not None and incumbent_exact_match is None:
        return True
    if candidate_exact_match is None and incumbent_exact_match is not None:
        return False

    if candidate.validation_loss != incumbent.validation_loss:
        return candidate.validation_loss < incumbent.validation_loss
    return candidate.epoch < incumbent.epoch
