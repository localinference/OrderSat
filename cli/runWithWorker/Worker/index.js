import { parentPort } from 'worker_threads'
import { unpackArchive } from './jobs/unpackArchive/index.js'
import { getTextFromImage } from './jobs/getTextFromImage/index.js'
import { pdfToImages } from './jobs/pdfToImages/index.js'

const jobs = {
  pdfToImages,
  unpackArchive,
  getTextFromImage,
}

parentPort.on('message', async (data) => {
  await runJob(data)
})

async function runJob({ id, job, params }) {
  try {
    const jobRun = jobs[job]
    const result = await jobRun(params)
    parentPort.postMessage({ id, result })
  } catch (err) {
    parentPort.postMessage({
      id,
      error: {
        message: err?.message ?? String(err),
        name: err?.name ?? 'Error',
        stack: err?.stack,
      },
    })
  }
}
