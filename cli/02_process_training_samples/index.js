import { getArgs } from '../utils/getArgs/index.js'
import FastGlob from 'fast-glob'
import { getOutputPathsFromInputPaths } from './getOutputPathsFromInputPaths/index.js'
const t0 = performance.now()

const sampleBase = './src/02_training_samples'
const outputSamplePath = sampleBase + '/outputs'
const tokenizerBase = './src/03_tokenizers'

try {
  /*************************************************/
  const { languages } = getArgs()
  for (const language of languages) {
    const languageOutputPath = `${outputSamplePath}/${language}`

    const outputFileNames = await FastGlob.async(
      `${languageOutputPath}/*.jsonld`
    )
    const inputFileNames = getOutputPathsFromInputPaths(outputFileNames)
    console.log('[OUTPUTS]', outputFileNames)
    console.log('[INPUTS]', inputFileNames)
  }
  /*************************************************/
} catch (err) {
  console.error(err)
}
