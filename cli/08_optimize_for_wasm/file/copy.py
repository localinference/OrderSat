def copy_support_artifacts(paths: QuantizationPaths) -> None:
    shutil.copy2(paths.source_tokenizer_model_path, paths.quantized_tokenizer_model_path)
    shutil.copy2(paths.source_tokenizer_vocab_path, paths.quantized_tokenizer_vocab_path)
    if paths.source_metrics_path.exists():
        shutil.copy2(paths.source_metrics_path, paths.quantized_metrics_path)

