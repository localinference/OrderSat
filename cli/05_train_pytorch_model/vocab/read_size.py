import pathlib

def read_vocab_size(vocab_path: pathlib.Path) -> int:
    if not vocab_path.exists():
        raise SystemExit(f"Tokenizer vocab does not exist: {vocab_path}")
    if not vocab_path.is_file():
        raise SystemExit(f"Tokenizer vocab path is not a file: {vocab_path}")

    with vocab_path.open("r", encoding="utf8") as handle:
        size = sum(1 for line in handle if line.strip())

    if size < 3:
        raise SystemExit(
            f"Tokenizer vocab looks invalid or too small: {vocab_path}"
        )

    return size