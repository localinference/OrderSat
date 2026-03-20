import * as ort from 'onnxruntime-web'
import { cleanText } from '@sctg/sentencepiece-js'
import {
  createTokenizer,
  createInferenceSession,
  GPUAccelerationSupported,
} from '@localinference/utils'
import { tokenizerENG, cpuModelENG, gpuModelENG, model } from '../dist/index.js'

const BOS_TOKEN_ID = 1
const EOS_TOKEN_ID = 2

const sample = `- 3
y TRADER JOE'S
2001 Greenville Ave
Dallas TX 75206
Store #403 - (469) 334-0614
OPEN 8:00AM TO 9:00PM DAILY
R-CARROTS SHREDDED 10 0Z 1.29
R-CUCUMBERS PERSIAN 1 LB 1.99
TOMATOES CRUSHED NO SALT 1.59
TOMATOES WHOLE NO SALT W/BASIL 1.59
ORGANIC OLD_FASHIONED OATMEAL ~~ 2.69
MINI-PEARL TOMATOES. . 2.49
PKG SHREDDED MOZZARELLA LITET 3.9
EGGS 1 DOZ ORGANIC BROWN. 3.79
BEANS GARBANZO 0.89
SPROUTED CA STYLE Zea
A-AVOCADOS HASS BAG ACT 2:39
A-APPLE BAG JAZZ 2 |B gr
A-PEPPER BELL EACH XL RED 0.99
GROCERY NON TAXABLE 0.98
260.49
BANANAS ORGANIC 0.87
3kA 6 0.29/EA
CREAMY SALTED PEANUT BUT TER 2.49
WHL WHT PITA BREAD 1.69
GROCERY NON TAXABLE 1.38
260.69
SUBTOTAL $38.68
TOTAL $38.68
CASH $40.00
CHANGE $1.32
ITEMS 22 Higgins, Ryan
06-28-2014 12:34PM 0403 04 1346 4683
THANK YOU FOR SHOPPING AT
TRADER JOE'S
www. trader joes .com
`

const input = cleanText(sample.normalize('NFKC'))

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

const tokenizer = await createTokenizer(await tokenizerENG())
const session = await createInferenceSession(
  await (
    GPUAccelerationSupported()
      ? async () => await gpuModelENG()
      : async () => await cpuModelENG()
  )()
)

const tokenIds = tokenizer.encodeIds(input)
const attentionMask = tokenIds.map(() => 1)
const decoderTokenIds = [BOS_TOKEN_ID]

while (true) {
  const outputs = await session.run({
    input_ids: toInt64Tensor(tokenIds, [1, tokenIds.length]),
    attention_mask: toInt64Tensor(attentionMask, [1, attentionMask.length]),
    decoder_input_ids: toInt64Tensor(decoderTokenIds, [
      1,
      decoderTokenIds.length,
    ]),
  })

  const nextTokenId = getNextTokenId(outputs.logits)
  if (nextTokenId === EOS_TOKEN_ID) {
    break
  }

  decoderTokenIds.push(nextTokenId)
}

const outputTokenIds = decoderTokenIds.slice(1)
const outputText = tokenizer.decodeIds(outputTokenIds)

console.log({
  inputLength: input.length,
  tokenCount: tokenIds.length,
  outputTokenCount: outputTokenIds.length,
  outputTokenIds,
  outputText,
})
