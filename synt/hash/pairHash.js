import crypto from 'crypto'

export function pairHash(inputText, outputStableText) {
  return crypto
    .createHash('sha384')
    .update(inputText)
    .update('\n')
    .update(outputStableText)
    .digest('base64url')
}
