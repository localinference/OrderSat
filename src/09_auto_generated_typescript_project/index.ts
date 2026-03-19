import { InferenceSession } from 'onnxruntime-web/all'

export async function prepareSaT(model: Uint8Array) {
  return await InferenceSession.create(model, {
    executionProviders: ['webnn', 'webgpu', 'webgl', 'wasm'],
  })
}

import { SentencePieceProcessor } from '@sctg/sentencepiece-js'

export async function prepareTokenizer(
  model: string
): Promise<SentencePieceProcessor> {
  const tokenizer = new SentencePieceProcessor()
  await tokenizer.loadFromB64StringModel(model)
  return tokenizer
}
