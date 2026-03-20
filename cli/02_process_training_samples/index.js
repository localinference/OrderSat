#!/usr/bin/env node

import { getArgs } from '../utils/getArgs/index.js'
import FastGlob from 'fast-glob'
import { getInputPathFromOutputPath } from './getInputPathFromOutputPath/index.js'
import fs from 'fs/promises'
import { cleanWhitespace } from '../utils/cleanWhiteSpace/index.js'

const outputSamplePath = './src/02_training_samples/outputs'
const tokenizerPath = './src/03_tokenizers'

function parseJson(json, jsonPath) {
  try {
    return JSON.parse(json)
  } catch (error) {
    throw new Error(
      `Invalid JSON output sample at ${jsonPath}: ${error?.message ?? String(error)}`
    )
  }
}

try {
  const { languages } = getArgs()

  for (const language of languages) {
    const languageT0 = performance.now()
    const jsonlLines = []
    const languageOutputPath = `${outputSamplePath}/${language}`
    const languageTokenizerPath = `${tokenizerPath}/${language}`
    const languageTokenizerCorpusPath = `${tokenizerPath}/${language}/corpus.jsonl`

    const outputFileNames = (
      await FastGlob(`${languageOutputPath}/*.jsonld`)
    ).sort((a, b) => a.localeCompare(b))

    if (outputFileNames.length === 0) {
      throw new Error(
        `No output samples found for language "${language}" at ${languageOutputPath}`
      )
    }

    await fs.stat(languageTokenizerPath)

    for (const outputPath of outputFileNames) {
      const inputPath = getInputPathFromOutputPath(outputPath)

      let [input, output] = await Promise.all([
        fs.readFile(inputPath, { encoding: 'utf-8' }),
        fs.readFile(outputPath, { encoding: 'utf-8' }),
      ])

      if (!input || !output) continue

      input = cleanWhitespace(input)
      if (!input) continue

      output = parseJson(output, outputPath)

      jsonlLines.push(JSON.stringify({ input, output }))
    }

    await fs.writeFile(
      languageTokenizerCorpusPath,
      jsonlLines.length ? `${jsonlLines.join('\n')}\n` : '',
      {
        encoding: 'utf-8',
      }
    )
    console.log(
      `Created tokenizer corpus at ${languageTokenizerCorpusPath} from ${jsonlLines.length} {input+output} samples for language "${language}" in ${Math.round(performance.now() - languageT0)} milliseconds.`
    )
  }
} catch (err) {
  console.error(err)
  process.exit(1)
}
