export function parseCorpusLine(line, lineNumber) {
  let parsed
  try {
    parsed = JSON.parse(line)
  } catch (error) {
    throw new Error(
      `Invalid JSONL at line ${lineNumber}: ${error?.message ?? String(error)}`
    )
  }

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`Expected JSON object at line ${lineNumber}`)
  }
  if (typeof parsed.input !== 'string') {
    throw new Error(`Expected string input at line ${lineNumber}`)
  }
  if (!Object.prototype.hasOwnProperty.call(parsed, 'output')) {
    throw new Error(`Missing output at line ${lineNumber}`)
  }

  return parsed
}
