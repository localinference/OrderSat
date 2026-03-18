import fs from 'fs/promises'
export async function writeJsonl(filePath, records) {
  const lines = records.map((record) => JSON.stringify(record))
  const content = lines.length === 0 ? '' : `${lines.join('\n')}\n`
  await fs.writeFile(filePath, content)
}
