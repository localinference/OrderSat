import { join } from 'path'
import { cleanText } from '@sctg/sentencepiece-js'
import {
  toBase64UrlString,
  toString,
  fromString,
} from '@sovereignbase/bytecodec'
import { ensurePath } from '../../utils/ensurePath/index.js'
import { writeFile } from 'fs/promises'
import { outputRoot } from '../index.js'

export async function writeUniqueToDest(textBuffer, language) {
  const cleanedString = cleanText(toString(textBuffer).normalize('NFKC'))

  if (cleanedString.trim().length === 0) {
    console.log(
      `[Output] Skipping empty cleaned output for language "${language}".\n`
    )
    return
  }

  const digest = await crypto.subtle.digest(
    'SHA-384',
    fromString(cleanedString)
  )

  const fileName = `${toBase64UrlString(new Uint8Array(digest))}.txt`
  const languageDestinationPath = await ensurePath(join(outputRoot, language))

  const path = join(languageDestinationPath, fileName)

  try {
    await writeFile(path, cleanedString, {
      encoding: 'utf8',
      flag: 'wx',
    })
    console.log(`[Output] Writing "${fileName}".\n`)
  } catch (err) {
    if (err?.code !== 'EEXIST') {
      throw err
    }
    console.log(`[Output] Skipping existing output "${fileName}".\n`)
  }
}
