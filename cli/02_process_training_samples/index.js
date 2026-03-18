import { getArgs } from '../utils/getArgs/index.js'
import FastGlob from 'fast-glob'
import { getInputPathFromOutputPath } from './getInputPathFromOutputPath/index.js'
import fs from 'fs/promises'
import { cleanWhitespace } from '../utils/cleanWhiteSpace/index.js'

const t0 = performance.now()
const outputSamplePath = './src/02_training_samples/outputs'
const tokenizerPath = './src/03_tokenizers'

try {
  const { languages } = getArgs()

  for (const language of languages) {
    let jsonl = ''
    const languageOutputPath = `${outputSamplePath}/${language}`
    const languageTokenizerCorpusPath = `${tokenizerPath}/${language}/corpus.jsonl`

    const outputFileNames = await FastGlob(`${languageOutputPath}/*.jsonld`)

    for (const outputPath of outputFileNames) {
      const inputPath = getInputPathFromOutputPath(outputPath)

      let input = await fs.readFile(inputPath, { encoding: 'utf-8' })
      let output = await fs.readFile(outputPath, { encoding: 'utf-8' })

      if (!input || !output) continue

      input = cleanWhitespace(input)
      output = cleanWhitespace(output)

      jsonl += `${JSON.stringify({ input, output })}\n`
    }

    await fs.writeFile(languageTokenizerCorpusPath, jsonl, {
      encoding: 'utf-8',
    })
    console.log(
      `Created tokenizer corpus for language "${language}" in ${Math.round(performance.now() - t0)} milliseconds`
    )
  }
} catch (err) {
  console.error(err)
}
