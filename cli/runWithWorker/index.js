import { Worker } from 'worker_threads'
import os from 'os'

export const availableParallelism = os.availableParallelism()

const workerPool = []
let lastWorkerIndex = -1

for (let i = 0; i < availableParallelism; i++) {
  workerPool.push(new Worker('./cli/runWithWorker/Worker/index.js'))
}

console.log(
  `Extracting input samples with up to ${availableParallelism} threads.\n`
)

export function runWithWorker(jobName, jobParams) {
  try {
    lastWorkerIndex = (lastWorkerIndex + 1) % availableParallelism
    const worker = workerPool[lastWorkerIndex]

    return new Promise((resolve) => {
      const id = crypto.randomUUID()

      const handleMessage = (data) => {
        if (data.id === id) {
          worker.off('message', handleMessage)
          resolve(data.result)
        }
      }

      worker.on('message', handleMessage)
      worker.postMessage({ id, job: jobName, params: jobParams })
    })
  } catch (err) {
    throw err
  }
}
