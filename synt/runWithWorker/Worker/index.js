import { parentPort } from 'worker_threads'
import { generateBatch } from './jobs/generateBatch/index.js'

const jobHandlers = {
  generateBatch,
}

parentPort.on('message', async (message) => {
  try {
    const handler = jobHandlers[message.job]
    if (!handler) {
      throw new Error(`Unknown worker job "${message.job}".`)
    }

    const result = await handler(message.params)
    parentPort.postMessage({
      id: message.id,
      result,
    })
  } catch (error) {
    parentPort.postMessage({
      id: message.id,
      error: {
        name: error?.name ?? 'Error',
        message: error?.message ?? String(error),
        stack: error?.stack ?? '',
      },
    })
  }
})
