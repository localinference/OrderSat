#!/usr/bin/env node
import fs from 'fs/promises'
import path from 'path'
import process from 'process'

import { getArgs } from '../utils/getArgs/index.js'
import { tokenizeSample } from './tokenizeSample/index.js'
import { summarizeLengths } from './summarizeLengths/index.js'
import { loadCorpusLines } from './readCorpusLines/index.js'
import { parseCorpusLine } from './parseCorpusLine/index.js'
import { splitSamples } from './splitSamples/index.js'
import { loadTokenizer } from './loadTokenizer/index.js'
import { writeJsonl } from './writeJSONL/index.js'

const { languages } = getArgs()

const DEFAULT_OPTIONS = {
  language: languages[0],
  tokenizers: 'src/03_tokenizers',
  datasets: 'src/04_training_datasets',
  corpusName: 'corpus.jsonl',
  modelName: 'tokenizer.model',
  validationRatio: 0.25,
  validationCount: null,
}

async function main() {
  const options = DEFAULT_OPTIONS
  const tokenizersRoot = path.resolve(process.cwd(), options.tokenizers)
  const datasetsRoot = path.resolve(process.cwd(), options.datasets)
  const languageRoot = path.join(tokenizersRoot, options.language)
  const corpusPath = path.join(languageRoot, options.corpusName)
  const modelPath = path.join(languageRoot, options.modelName)
  const outputRoot = path.join(datasetsRoot, options.language)

  const [corpusLines] = await Promise.all([
    loadCorpusLines(corpusPath),
    fs.access(modelPath),
  ])

  const processor = await loadTokenizer(modelPath)
  const tokenizedSamples = corpusLines.map((line, index) => {
    const lineNumber = index + 1
    const parsed = parseCorpusLine(line, lineNumber)
    return tokenizeSample(processor, parsed, lineNumber)
  })

  const { train, validation } = splitSamples(tokenizedSamples, options)
  const stats = {
    language: options.language,
    corpusPath,
    modelPath,
    sampleCount: tokenizedSamples.length,
    trainCount: train.length,
    validationCount: validation.length,
    inputLengths: summarizeLengths(
      tokenizedSamples.map((sample) => sample.input_length)
    ),
    labelLengths: summarizeLengths(
      tokenizedSamples.map((sample) => sample.label_length)
    ),
  }

  await fs.mkdir(outputRoot, { recursive: true })
  await Promise.all([
    writeJsonl(path.join(outputRoot, 'train.jsonl'), train),
    writeJsonl(path.join(outputRoot, 'validation.jsonl'), validation),
    fs.writeFile(
      path.join(outputRoot, 'stats.json'),
      `${JSON.stringify(stats, null, 2)}\n`
    ),
  ])

  console.log(`Done. Tokenized dataset written to ${outputRoot}`)
  console.log(`Samples: ${tokenizedSamples.length}`)
  console.log(`Train: ${train.length}`)
  console.log(`Validation: ${validation.length}`)
  console.log(`Input max length: ${stats.inputLengths.max}`)
  console.log(`Label max length: ${stats.labelLengths.max}`)
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
