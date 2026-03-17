import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'
import { toCompressed, toBase64UrlString } from '@z-base/bytecodec'

const js = String.raw

const modelPath = './models/quantized_models/eng/model.int4.onnx'
const modelDataPath = './models/quantized_models/eng/model.int4.onnx.data'
const tokenizerModelPath = './models/quantized_models/eng/tokenizer.model'

const outPath = './src/models/index.ts'

const [model, modelData, tokenizerModel] = await Promise.all([
  readFile(modelPath),
  readFile(modelDataPath),
  readFile(tokenizerModelPath),
])

const modelBase64Url = JSON.stringify(
  toBase64UrlString(await toCompressed(model))
)
const modelDataBase64Url = JSON.stringify(
  toBase64UrlString(await toCompressed(modelData))
)
const tokenizerModelBase64 = JSON.stringify(tokenizerModel.toString('base64'))

const ts = js`
import * as ort from 'onnxruntime-web'
import { SentencePieceProcessor } from '@sctg/sentencepiece-js'
import { fromCompressed, fromBase64UrlString } from '@z-base/bytecodec'

export async function createInferenceSession(): Promise<ort.InferenceSession> {
  return ort.InferenceSession.create(
    await fromCompressed(
      fromBase64UrlString(${modelBase64Url})
    ),
    {
      externalData: [
        {
          path: 'model.int4.onnx.data',
          data: await fromCompressed(
            fromBase64UrlString(${modelDataBase64Url})
          ),
        },
      ],
    }
  )
}

export async function createTokenProcessor(): Promise<SentencePieceProcessor> {
  const tokenProcessor = new SentencePieceProcessor()
  await tokenProcessor.loadFromB64StringModel(${tokenizerModelBase64})
  return tokenProcessor
}
`.trimStart()

await mkdir(dirname(outPath), { recursive: true })
await writeFile(outPath, ts, 'utf8')
