import { parseArgs } from 'node:util'
import fs from 'node:fs/promises'

import { validateStructure } from '../utils/validateStructure/index.js'

const { positionals } = parseArgs({
  allowPositionals: true,
})

const inputPath = positionals[0]

if (!inputPath) {
  console.error(
    'Usage: node cli/validateStructure/batch.js <predictions.jsonl>'
  )
  process.exit(1)
}

const raw = await fs.readFile(inputPath, { encoding: 'utf-8' })
const lines = raw.split(/\r?\n/).filter(Boolean)
const results = []

for (const [index, line] of lines.entries()) {
  let parsed
  try {
    parsed = JSON.parse(line)
  } catch (error) {
    throw new Error(
      `Invalid JSONL row at ${inputPath}:${index + 1}: ${error?.message ?? String(error)}`
    )
  }

  const sampleId = parsed?.sample_id
  const outputText = parsed?.output_text

  if (typeof sampleId !== 'string') {
    throw new Error(`Missing string sample_id at ${inputPath}:${index + 1}`)
  }
  if (typeof outputText !== 'string') {
    throw new Error(`Missing string output_text at ${inputPath}:${index + 1}`)
  }

  const validation = await validateStructure(outputText)
  if (validation === true) {
    results.push({
      sample_id: sampleId,
      valid_structure: true,
      issues: [],
    })
    continue
  }

  results.push({
    sample_id: sampleId,
    valid_structure: false,
    issues: Array.isArray(validation) ? validation : [validation],
  })
}

console.log(
  JSON.stringify(
    {
      sample_count: results.length,
      results,
    },
    null,
    2
  )
)
