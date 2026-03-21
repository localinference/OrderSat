import fs from 'fs/promises'
import { join } from 'path'
import { ensurePath } from '../../cli/utils/ensurePath/index.js'

export async function writeSamplePair({
  fileStem,
  inputRoot,
  outputRoot,
  language,
  inputText,
  outputText,
}) {
  const inputDirectory = await ensurePath(join(inputRoot, language))
  const outputDirectory = await ensurePath(join(outputRoot, language))
  const inputPath = join(inputDirectory, `${fileStem}.txt`)
  const outputPath = join(outputDirectory, `${fileStem}.jsonld`)

  const [inputStatus, outputStatus] = await Promise.all([
    writeIfMissing(inputPath, `${inputText}\n`),
    writeIfMissing(outputPath, `${outputText}\n`),
  ])

  return {
    written: inputStatus.written || outputStatus.written,
  }
}

async function writeIfMissing(path, text) {
  try {
    await fs.writeFile(path, text, {
      encoding: 'utf-8',
      flag: 'wx',
    })

    return { written: true }
  } catch (error) {
    if (error?.code === 'EEXIST') {
      return { written: false }
    }

    throw error
  }
}
