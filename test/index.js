import * as ort from 'onnxruntime-web'
import { cleanText } from '@sctg/sentencepiece-js'
import {
  createInferenceSession,
  createTokenizer,
  GPUAccelerationSupported,
} from '@localinference/utils'
import {
  cpuModelENG,
  gpuModelENG,
  modelInfoENG,
  tokenizerENG,
} from '../dist/index.js'

const sample = `order id: 10869 shipping details: ship name: seven seas imports ship address: 90 wadhurst rd. ship city: london ship region: british isles ship postal code: ox15 4nb ship country: uk customer details: customer id: seves customer name: seven seas imports employee details: employee name: steven buchanan shipper details: shipper id: 1 shipper name: speedy express order details: order date: 2018-02-04 shipped date: 2018-02-09 products: product: chai quantity: 40 unit price: 18.0 total: 720.0 product: queso cabrales quantity: 10 unit price: 21.0 total: 210.0 product: tunnbrod quantity: 50 unit price: 9.0 total: 450.0 product: scottish longbreads quantity: 20 unit price: 12.5 total: 250.0 total price: total price: 1630.0`

function toInt64Tensor(values, dims) {
  return new ort.Tensor('int64', BigInt64Array.from(values, BigInt), dims)
}

function argmax(values) {
  let bestIndex = 0
  let bestValue = Number.NEGATIVE_INFINITY

  for (let index = 0; index < values.length; index += 1) {
    if (values[index] > bestValue) {
      bestValue = values[index]
      bestIndex = index
    }
  }

  return bestIndex
}

function getNextTokenId(logits) {
  const [, targetLength, vocabSize] = logits.dims
  const offset = (targetLength - 1) * vocabSize
  const stepLogits = logits.data.subarray(offset, offset + vocabSize)

  return argmax(stepLogits)
}

function normalizeInput(text) {
  return cleanText(text.normalize('NFKC'))
}

function encodeInput(text, tokenizer, modelInfo) {
  const inputText = normalizeInput(text)
  const inputTokenIds = tokenizer.encodeIds(inputText)

  if (inputTokenIds.length > modelInfo.maxSourcePositions) {
    throw new Error(
      [
        `Input exceeds model maxSourcePositions.`,
        `tokenCount=${inputTokenIds.length}`,
        `maxSourcePositions=${modelInfo.maxSourcePositions}`,
        `The test harness does not truncate inputs automatically.`,
      ].join(' ')
    )
  }

  return {
    inputText,
    inputTokenIds,
    attentionMask: inputTokenIds.map(() => 1),
  }
}

async function loadRuntime() {
  const useGPU = GPUAccelerationSupported()
  const loader = useGPU ? gpuModelENG : cpuModelENG
  const modelBytes = await loader()
  const session = await createInferenceSession(modelBytes)

  return {
    runtime: useGPU ? 'gpu' : 'cpu',
    session,
  }
}

async function greedyDecode({
  session,
  inputIdsTensor,
  attentionMaskTensor,
  modelInfo,
}) {
  const decoderTokenIds = [modelInfo.bosTokenId]

  while (true) {
    const outputs = await session.run({
      input_ids: inputIdsTensor,
      attention_mask: attentionMaskTensor,
      decoder_input_ids: toInt64Tensor(decoderTokenIds, [
        1,
        decoderTokenIds.length,
      ]),
    })

    const nextTokenId = getNextTokenId(outputs.logits)
    if (nextTokenId === modelInfo.eosTokenId) {
      return decoderTokenIds.slice(1)
    }

    if (decoderTokenIds.length >= modelInfo.maxTargetPositions) {
      throw new Error(
        [
          `Model failed to emit EOS within maxTargetPositions.`,
          `decoderLength=${decoderTokenIds.length}`,
          `maxTargetPositions=${modelInfo.maxTargetPositions}`,
        ].join(' ')
      )
    }

    decoderTokenIds.push(nextTokenId)
  }
}

const tokenizer = await createTokenizer(await tokenizerENG())
const { runtime, session } = await loadRuntime()
const { inputText, inputTokenIds, attentionMask } = encodeInput(
  sample,
  tokenizer,
  modelInfoENG
)

const inputIdsTensor = toInt64Tensor(inputTokenIds, [1, inputTokenIds.length])
const attentionMaskTensor = toInt64Tensor(attentionMask, [
  1,
  attentionMask.length,
])
const outputTokenIds = await greedyDecode({
  session,
  inputIdsTensor,
  attentionMaskTensor,
  modelInfo: modelInfoENG,
})
const outputText = tokenizer.decodeIds(outputTokenIds)

console.log({
  runtime,
  modelInfo: modelInfoENG,
  inputLength: inputText.length,
  inputTokenCount: inputTokenIds.length,
  outputTokenCount: outputTokenIds.length,
  outputText,
})
