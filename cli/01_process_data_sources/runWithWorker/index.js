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
  `[Pool] Worker pool ready with up to ${availableParallelism} thread(s).\n`
)

export function runWithWorker(jobName, jobParams) {
  if (isClosing) {
    return Promise.reject(new Error('Worker pool is closing.'))
  }

  return new Promise((resolve, reject) => {
    const id = crypto.randomUUID()

    pendingJobs.push({
      id,
      jobName,
      jobParams,
      reject,
      resolve,
    })
    console.log(
      `[Pool] Queued ${formatJob(jobName, id)}. Pending: ${pendingJobs.length}. Running: ${runningJobs.size}.\n`
    )
    drainQueue()
  })
}

export async function closeWorkerPool() {
  isClosing = true
  console.log(
    `[Pool] Closing worker pool. Pending: ${pendingJobs.length}. Running: ${runningJobs.size}.\n`
  )

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
  console.log(`[Pool] Worker pool termination finished.\n`)
}

function createWorker() {
  const worker = new Worker(new URL('./Worker/index.js', import.meta.url))
  console.log(`[Pool] Created worker ${worker.threadId}.\n`)

  worker.on('message', (data) => {
    const runningJob = runningJobs.get(worker)
    if (!runningJob || data.id !== runningJob.id) {
      return
    }

    runningJobs.delete(worker)
    console.log(
      `[Pool] Completed ${formatJob(runningJob.jobName, runningJob.id)} on worker ${worker.threadId}. Pending: ${pendingJobs.length}. Running: ${runningJobs.size}.\n`
    )
    if (!isClosing) {
      idleWorkers.push(worker)
      drainQueue()
    }

    if (data.error) {
      const error = new Error(data.error.message)
      error.name = data.error.name
      error.stack = data.error.stack
      console.log(
        `[Pool] ${formatJob(runningJob.jobName, runningJob.id)} failed on worker ${worker.threadId}: "${error}".\n`
      )
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
    console.log(
      `[Pool] Dispatching ${formatJob(job.jobName, job.id)} to worker ${worker.threadId}. Pending: ${pendingJobs.length}. Running: ${runningJobs.size}.\n`
    )
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

  console.log(
    `[Pool] Worker ${worker.threadId} failed: "${error}". Replacing worker.\n`
  )

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

function formatJob(jobName, id) {
  return `job "${jobName}" (${id.slice(0, 8)})`
}
