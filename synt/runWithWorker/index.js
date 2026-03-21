import { Worker } from 'worker_threads'

export function createWorkerPool(maxWorkers) {
  const workerPool = []
  const idleWorkers = []
  const pendingJobs = []
  const runningJobs = new Map()
  let isClosing = false

  for (let index = 0; index < maxWorkers; index += 1) {
    createWorker()
  }

  return {
    run,
    close,
  }

  function run(jobName, jobParams) {
    if (isClosing) {
      return Promise.reject(new Error('Worker pool is closing.'))
    }

    return new Promise((resolve, reject) => {
      const id = crypto.randomUUID()
      pendingJobs.push({
        id,
        jobName,
        jobParams,
        resolve,
        reject,
      })
      drainQueue()
    })
  }

  async function close() {
    isClosing = true

    const queuedJobs = pendingJobs.splice(0)
    for (const queuedJob of queuedJobs) {
      queuedJob.reject(new Error('Worker pool closed before job started.'))
    }

    const workers = workerPool.splice(0)
    idleWorkers.length = 0
    runningJobs.clear()
    await Promise.all(workers.map((worker) => worker.terminate()))
  }

  function createWorker() {
    const worker = new Worker(new URL('./Worker/index.js', import.meta.url))

    worker.on('message', (message) => {
      const runningJob = runningJobs.get(worker)
      if (!runningJob || runningJob.id !== message.id) {
        return
      }

      runningJobs.delete(worker)
      if (!isClosing) {
        idleWorkers.push(worker)
        drainQueue()
      }

      if (message.error) {
        const error = new Error(message.error.message)
        error.name = message.error.name
        error.stack = message.error.stack
        runningJob.reject(error)
        return
      }

      runningJob.resolve(message.result)
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
    const runningJob = runningJobs.get(worker)
    runningJobs.delete(worker)

    const workerIndex = workerPool.indexOf(worker)
    if (workerIndex !== -1) {
      workerPool.splice(workerIndex, 1)
    }

    const idleIndex = idleWorkers.indexOf(worker)
    if (idleIndex !== -1) {
      idleWorkers.splice(idleIndex, 1)
    }

    if (runningJob) {
      runningJob.reject(error)
    }

    if (!isClosing) {
      createWorker()
      drainQueue()
    }
  }
}
