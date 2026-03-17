import { Worker } from 'worker_threads'
import os from 'os'

export const availableParallelism = os.availableParallelism()

const workerPool = []
let lastWorkerIndex = -1

for (let i = 0; i < availableParallelism; i++) {
  workerPool.push(new Worker(new URL('./Worker/index.js', import.meta.url)))
}

console.log(
  `Extracting input samples with up to ${availableParallelism} threads.\n`
)

export function runWithWorker(jobName, jobParams) {
  lastWorkerIndex = (lastWorkerIndex + 1) % availableParallelism
  const worker = workerPool[lastWorkerIndex]

  return new Promise((resolve, reject) => {
    const id = crypto.randomUUID()

    const cleanup = () => {
      worker.off('message', handleMessage)
      worker.off('error', handleError)
    }

    const handleMessage = (data) => {
      if (data.id === id) {
        cleanup()
        if (data.error) {
          const error = new Error(data.error.message)
          error.name = data.error.name
          error.stack = data.error.stack
          reject(error)
          return
        }
        resolve(data.result)
      }
    }

    const handleError = (error) => {
      cleanup()
      reject(error)
    }

    worker.on('message', handleMessage)
    worker.on('error', handleError)
    worker.postMessage({ id, job: jobName, params: jobParams })
  })
}

export async function closeWorkerPool() {
  await Promise.all(workerPool.map((worker) => worker.terminate()))
}
