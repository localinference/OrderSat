import fs from 'fs/promises'
export async function loadCorpusLines(corpusPath) {
  const raw = await fs.readFile(corpusPath, 'utf8')
  return raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
}
