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

const sample = `<orderDocument>
<seller name="Wild Pub Global" phone="+64 4 889 6829" email="wild85@sales.example.com">
<address>36, Constable Street, Wellington, Wellington, 1311, New Zealand</address>
</seller>
<order id="ORD89918193" placedAt="06/05/2025 12:22" payment="Debit Card" status="Delivered"> <customer>Svetlana Z. Strickland</customer>
<items><item><name>Media Player</name><quantity>4</quantity><unitPrice>309.01 NZD</unitPrice><linePrice>1236.04 NZD</linePrice></item><item><name>Laundry Hamper</name><quantity>2</quantity><unitPrice>NZD 160.7</unitPrice><linePrice>NZ$321.4</linePrice></item><item><name>Rail Pass</name><quantity>2</quantity><unitPrice>NZD 257,19</unitPrice><linePrice>NZ$514.38</linePrice></item><item><name>Digital Camera</name><quantity>1</quantity><unitPrice>NZD 416.17</unitPrice><linePrice>NZD 416,17</linePrice></item></items>
<totals subtotal="NZ$2487.99" tax="NZ$1.17" total="NZ$5O.93" />
<delivery provider="Turbo Freight Global" tracking="1Z242F9KTR9D2AW0J0" shippedDate="2025-06-07">
<address>164, 5tratford Road, Birmingham, England, MF4 1LB, United Kingdom</address>
</delivery>
</order>
</orderDocument>
`

function toInt32Tensor(values, dims) {
  return new ort.Tensor('int32', Int32Array.from(values), dims)
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
      decoder_input_ids: toInt32Tensor(decoderTokenIds, [
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

const inputIdsTensor = toInt32Tensor(inputTokenIds, [1, inputTokenIds.length])
const attentionMaskTensor = toInt32Tensor(attentionMask, [
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
