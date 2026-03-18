import { getArgs } from '../utils/getArgs/index.js'
import FastGlob from 'fast-glob'
const t0 = performance.now()

const sampleBase = './src/02_training_samples'

const inputSamplePath = sampleBase + '/inputs'
const outputSamplePath = sampleBase + '/outputs'

const tokenizerBase = './src/03_tokenizers'

try {
  /*************************************************/
  const { languages } = getArgs()
  for (const language of languages) {
    const languageOutputPath = `${outputSamplePath}/${language}`
    console.log(languageOutputPath)
    const outputFileNames = await FastGlob.async('/*.jsonld', {
      cwd: languageOutputPath,
      dot: true,
      onlyFiles: true,
    })
    console.log(outputFileNames)
  }
  /*************************************************/
} catch (err) {
  console.error(err)
}
