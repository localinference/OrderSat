
def validate_quantized_model(model_path: pathlib.Path, source_config: dict[str, Any]) -> dict[str, Any]:
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    input_names = source_config.get(
        "inputNames",
        ["input_ids", "attention_mask", "decoder_input_ids"],
    )
    output_names = source_config.get("outputNames", ["logits"])
    token_ids = source_config.get("tokenIds", {})
    bos_id = int(token_ids.get("bos", 1))
    vocab_size = int(source_config.get("model", {}).get("vocabSize", 0))
    validation_cases: list[dict[str, Any]] = []

    for case in build_validation_cases(source_config):
        source_length = int(case["source_length"])
        target_length = int(case["target_length"])
        ort_inputs = {
            input_names[0]: np.ones((1, source_length), dtype=np.int64),
            input_names[1]: np.ones((1, source_length), dtype=np.int64),
            input_names[2]: np.full((1, target_length), bos_id, dtype=np.int64),
        }
        outputs = session.run(output_names, ort_inputs)
        logits = outputs[0]

        expected_shape = (1, target_length)
        actual_prefix_shape = tuple(logits.shape[:2])
        if actual_prefix_shape != expected_shape:
            raise SystemExit(
                f"Quantized model output shape mismatch for case '{case['name']}': "
                f"expected prefix {expected_shape}, got {actual_prefix_shape}"
            )
        if vocab_size and logits.shape[-1] != vocab_size:
            raise SystemExit(
                f"Quantized model vocab axis mismatch for case '{case['name']}': "
                f"expected {vocab_size}, got {logits.shape[-1]}"
            )

        validation_cases.append(
            {
                "name": case["name"],
                "inputShape": {
                    input_names[0]: list(ort_inputs[input_names[0]].shape),
                    input_names[1]: list(ort_inputs[input_names[1]].shape),
                    input_names[2]: list(ort_inputs[input_names[2]].shape),
                },
                "outputShape": list(logits.shape),
            }
        )

    return {
        "inputs": [item.name for item in session.get_inputs()],
        "outputs": [item.name for item in session.get_outputs()],
        "providers": session.get_providers(),
        "cases": validation_cases,
    }
