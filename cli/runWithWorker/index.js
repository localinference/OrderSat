import { Worker } from 'worker_threads'
import os from 'os'

export const availableParallelism = os.availableParallelism()

const workerPool = []
const idleWorkers = []
const pendingJobs = []
const runningJobs = new Map()
let isClosing = false

for (let i = 0; i < availableParallelism; i++) {
  createWorker()
}

console.log(
  `Extracting input samples with up to ${availableParallelism} threads.\n`
)

export function runWithWorker(jobName, jobParams) {
  if (isClosing) {
    return Promise.reject(new Error('Worker pool is closing.'))
  }

  return new Promise((resolve, reject) => {
    pendingJobs.push({
      id: crypto.randomUUID(),
      jobName,
      jobParams,
      reject,
      resolve,
    })
    drainQueue()
  })
}

export async function closeWorkerPool() {
  isClosing = true

  const queuedJobs = pendingJobs.splice(0)
  for (const job of queuedJobs) {
    job.reject(new Error('Worker pool closed before queued job started.'))
  }

  const activeJobs = [...runningJobs.values()]
  runningJobs.clear()
  for (const job of activeJobs) {
    job.reject(new Error('Worker pool closed before active job completed.'))
  }

  idleWorkers.length = 0

  const workers = workerPool.splice(0)
  await Promise.all(workers.map((worker) => worker.terminate()))
}

function createWorker() {
  const worker = new Worker(new URL('./Worker/index.js', import.meta.url))

  worker.on('message', (data) => {
    const runningJob = runningJobs.get(worker)
    if (!runningJob || data.id !== runningJob.id) {
      return
    }

    runningJobs.delete(worker)
    if (!isClosing) {
      idleWorkers.push(worker)
      drainQueue()
    }

    if (data.error) {
      const error = new Error(data.error.message)
      error.name = data.error.name
      error.stack = data.error.stack
      runningJob.reject(error)
      return
    }

    runningJob.resolve(data.result)
  })

  worker.on('error', (error) => {
    failWorker(worker, error)
  })

  worker.on('exit', (code) => {
    if (!isClosing && code !== 0) {
      failWorker(worker, new Error(`Worker exited with code ${code}.`))
    }
  })

  workerPool.push(worker)
  idleWorkers.push(worker)
}

function drainQueue() {
  while (!isClosing && idleWorkers.length > 0 && pendingJobs.length > 0) {
    const worker = idleWorkers.shift()
    const job = pendingJobs.shift()

    runningJobs.set(worker, job)
    worker.postMessage({
      id: job.id,
      job: job.jobName,
      params: job.jobParams,
    })
  }
}

function failWorker(worker, error) {
  if (!removeWorker(worker)) {
    return
  }

  const runningJob = runningJobs.get(worker)
  if (runningJob) {
    runningJobs.delete(worker)
    runningJob.reject(error)
  }

  if (!isClosing) {
    createWorker()
    drainQueue()
  }
}

function removeWorker(worker) {
  const workerIndex = workerPool.indexOf(worker)
  if (workerIndex === -1) {
    return false
  }

  workerPool.splice(workerIndex, 1)

  const idleIndex = idleWorkers.indexOf(worker)
  if (idleIndex !== -1) {
    idleWorkers.splice(idleIndex, 1)
  }

  return true
}
