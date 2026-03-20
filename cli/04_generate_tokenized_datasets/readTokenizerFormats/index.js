import fs from 'fs/promises'
import path from 'path'

const MODEL_NAME = 'tokenizer.model'

export async function readTokenizerFormats(languageRoot) {
  const entries = await fs.readdir(languageRoot, { withFileTypes: true })
  const formats = []

  for (const entry of entries) {
    if (!entry.isDirectory()) continue

    const formatRoot = path.join(languageRoot, entry.name)
    const modelPath = path.join(formatRoot, MODEL_NAME)

    try {
      await fs.access(modelPath)
      formats.push({
        format: entry.name,
        modelPath,
      })
    } catch {
      continue
    }
  }

  return formats.sort((a, b) => a.format.localeCompare(b.format))
}
