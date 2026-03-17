import { writeFile } from 'fs/promises'
import { join } from 'path'
import { cleanText } from '@sctg/sentencepiece-js'
import { toBase64UrlString, toString, fromString } from '@z-base/bytecodec'

export async function writeUniqueToDest(textBuffer, language, destinationPath) {
  const cleanedString = cleanText(toString(textBuffer).normalize('NFKC'))

  const digest = await crypto.subtle.digest(
    'SHA-384',
    fromString(cleanedString)
  )

  const fileName = `${toBase64UrlString(new Uint8Array(digest))}.txt`

  console.log(`Writing "${fileName}".`)
  writeFile(join(destinationPath, language, fileName), cleanedString, {
    encoding: 'utf8',
  })
}
