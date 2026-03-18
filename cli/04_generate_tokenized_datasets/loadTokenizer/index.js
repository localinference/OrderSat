import { SentencePieceProcessor } from '@sctg/sentencepiece-js'

export async function loadTokenizer(modelPath) {
  const processor = new SentencePieceProcessor()
  await processor.load(modelPath)
  return processor
}
