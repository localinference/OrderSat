import pathlib

from arguments.parse import parse_args

try:
    import sentencepiece as spm
except ImportError as error:
    raise SystemExit(
        "sentencepiece is required to train tokenizer vocabs. "
        "Install it in your Python environment first."
    ) from error


DEFAULT_TOKENIZERS_ROOT = pathlib.Path("src/03_tokenizers")
DEFAULT_CORPUS_NAME = "corpus.jsonl"
DEFAULT_MODEL_PREFIX = "tokenizer"
DEFAULT_MAX_SENTENCE_LENGTH = 16384

args = parse_args()