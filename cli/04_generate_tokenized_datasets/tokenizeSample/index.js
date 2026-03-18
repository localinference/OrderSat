import crypto from 'crypto'

function stableJsonValue(value) {
  if (Array.isArray(value)) {
    return value.map((item) =>
      item === undefined ? null : stableJsonValue(item)
    )
  }
  if (value && typeof value === 'object') {
    const normalized = {}
    const keys = Object.keys(value).sort((a, b) => a.localeCompare(b))
    for (const key of keys) {
      const item = value[key]
      if (item === undefined) continue
      normalized[key] = stableJsonValue(item)
    }
    return normalized
  }
  return value
}

function stableStringify(value) {
  return JSON.stringify(stableJsonValue(value))
}

function hashSample(inputText, outputText) {
  return crypto
    .createHash('sha256')
    .update(inputText)
    .update('\n')
    .update(outputText)
    .digest('hex')
}

export function tokenizeSample(processor, parsed, lineNumber) {
  const inputText = parsed.input
  const outputText = stableStringify(parsed.output)
  const inputIds = processor.encodeIds(inputText)
  const labels = processor.encodeIds(outputText)

  return {
    sample_id: hashSample(inputText, outputText),
    source_line: lineNumber,
    input_text: inputText,
    output_text: outputText,
    input_ids: inputIds,
    attention_mask: inputIds.map(() => 1),
    labels,
    input_length: inputIds.length,
    label_length: labels.length,
  }
}
