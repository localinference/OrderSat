import { parentPort, workerData } from 'worker_threads'
import { createWorker } from 'tesseract.js'
try {
  const worker = await createWorker(workerData.language)
  parentPort.on('message', async (ev) => {
    const {
      data: { text },
    } = await worker.recognize(ev.data)
    parentPort.postMessage(text)
  })
} catch (error) {
  throw error
}
