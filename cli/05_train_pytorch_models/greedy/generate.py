import torch


def greedy_generate(
    model,
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
        dtype=input_ids.dtype,
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
