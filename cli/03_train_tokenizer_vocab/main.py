import pathlib

from args.parse import parse_args

try:
    import sentencepiece as spm
except ImportError as error:
    raise SystemExit(
        "sentencepiece is required to train tokenizer vocabs. "
        "Install it in your Python environment first."
    ) from error


TOKENIZERS_DIR = pathlib.Path("src/03_tokenizers")
CORPUS_NAME = "corpus.jsonl"
MODEL_PREFIX = "tokenizer"
MAX_SENTENCE_LENGTH = 16384


def main() -> None:
    args = parse_args()
    language_dir = TOKENIZERS_DIR / args.language
    input_path = language_dir / CORPUS_NAME
    output_path = language_dir / MODEL_PREFIX




if __name__ == "__main__":
    main()