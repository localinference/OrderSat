import Seq2SeqTransformer
def count_parameters(model: Seq2SeqTransformer) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
