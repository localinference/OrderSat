import { fromCompressed, fromBase64UrlString } from '@z-base/bytecodec'
type Runtime = 'WASM' | 'WebGPU'
export async function engModelLoader(runWith: Runtime) {
  const options = {
    WASM: '',
    WebGPU: '',
  }
  return {
    inference: await fromCompressed(fromBase64UrlString(options[runWith])),
    tokenizer: '',
  }
}
