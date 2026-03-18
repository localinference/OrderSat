def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load tokenized seq2seq datasets, pad them with a collator, and "
            "run a tiny random-init Transformer overfit experiment."
        )
    )
    parser.add_argument(
        "language_positional",
        nargs="?",
        help="Language directory under models/datasets and models/tokenizers.",
    )
    parser.add_argument(
        "--language",
        dest="language_flag",
        help="Language directory under models/datasets and models/tokenizers.",
    )
    parser.add_argument(
        "--datasets-root",
        default=str(DEFAULT_DATASETS_ROOT),
        help="Dataset root directory (default: models/datasets)",
    )
    parser.add_argument(
        "--tokenizers-root",
        default=str(DEFAULT_TOKENIZERS_ROOT),
        help="Tokenizer root directory (default: models/tokenizers)",
    )
    parser.add_argument(
        "--train-file",
        default=DEFAULT_TRAIN_FILE,
        help="Train split filename (default: train.jsonl)",
    )
    parser.add_argument(
        "--validation-file",
        default=DEFAULT_VALIDATION_FILE,
        help="Validation split filename (default: validation.jsonl)",
    )
    parser.add_argument(
        "--vocab-file",
        default=DEFAULT_VOCAB_FILE,
        help="Tokenizer vocab filename (default: tokenizer.vocab)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"AdamW learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help=f"AdamW weight decay (default: {DEFAULT_WEIGHT_DECAY})",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=DEFAULT_D_MODEL,
        help=f"Transformer embedding width (default: {DEFAULT_D_MODEL})",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=DEFAULT_NUM_HEADS,
        help=f"Attention heads (default: {DEFAULT_NUM_HEADS})",
    )
    parser.add_argument(
        "--num-encoder-layers",
        type=int,
        default=DEFAULT_NUM_ENCODER_LAYERS,
        help=f"Encoder layer count (default: {DEFAULT_NUM_ENCODER_LAYERS})",
    )
    parser.add_argument(
        "--num-decoder-layers",
        type=int,
        default=DEFAULT_NUM_DECODER_LAYERS,
        help=f"Decoder layer count (default: {DEFAULT_NUM_DECODER_LAYERS})",
    )
    parser.add_argument(
        "--ffn-dim",
        type=int,
        default=DEFAULT_FFN_DIM,
        help=f"Feed-forward width (default: {DEFAULT_FFN_DIM})",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT,
        help=f"Dropout rate (default: {DEFAULT_DROPOUT})",
    )
    parser.add_argument(
        "--bos-id",
        type=int,
        default=DEFAULT_BOS_ID,
        help=f"Beginning-of-sequence token id (default: {DEFAULT_BOS_ID})",
    )
    parser.add_argument(
        "--eos-id",
        type=int,
        default=DEFAULT_EOS_ID,
        help=f"End-of-sequence token id (default: {DEFAULT_EOS_ID})",
    )
    parser.add_argument(
        "--label-pad-id",
        type=int,
        default=DEFAULT_LABEL_PAD_ID,
        help=f"Loss ignore index for padded labels (default: {DEFAULT_LABEL_PAD_ID})",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=DEFAULT_MAX_INPUT_LENGTH,
        help=f"Input truncation length (default: {DEFAULT_MAX_INPUT_LENGTH})",
    )
    parser.add_argument(
        "--max-label-length",
        type=int,
        default=DEFAULT_MAX_LABEL_LENGTH,
        help=(
            "Label truncation length before appending EOS "
            f"(default: {DEFAULT_MAX_LABEL_LENGTH})"
        ),
    )
    parser.add_argument(
        "--max-generation-length",
        type=int,
        default=None,
        help="Optional greedy decoding length override.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_LOG_EVERY,
        help=f"Epoch logging interval (default: {DEFAULT_LOG_EVERY})",
    )
    parser.add_argument(
        "--exact-match-every",
        type=int,
        default=DEFAULT_EXACT_MATCH_EVERY,
        help=(
            "Greedy exact-match evaluation interval in epochs "
            f"(default: {DEFAULT_EXACT_MATCH_EVERY}, disabled)"
        ),
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=DEFAULT_GRAD_CLIP,
        help=f"Gradient clipping max norm (default: {DEFAULT_GRAD_CLIP})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, or mps (default: auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help=(
            "Stop when validation loss does not improve for this many epochs "
            f"(default: {DEFAULT_EARLY_STOPPING_PATIENCE})"
        ),
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=DEFAULT_EARLY_STOPPING_MIN_DELTA,
        help=(
            "Minimum validation loss improvement to reset patience "
            f"(default: {DEFAULT_EARLY_STOPPING_MIN_DELTA})"
        ),
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Override output directory (default: models/best_models/{language}).",
    )
    args = parser.parse_args()
    args.language = args.language_flag or args.language_positional

    if not args.language:
        parser.error("the following arguments are required: language")
    if args.language_flag and args.language_positional:
        parser.error("use either positional language or --language, not both")

    return args
