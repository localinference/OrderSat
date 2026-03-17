import { parentPort } from 'worker_threads'
import { unpackArchive } from './jobs/unpackArchive/index.js'
import { getTextFromImage } from './jobs/getTextFromImage/index.js'
import { writeUniqueToDest } from './jobs/writeUniqueToDest/index.js'
import { pdfToImage } from './jobs/pdfToImage/index.js'

const jobs = {
  pdfToImage,
  unpackArchive,
  getTextFromImage,
  writeUniqueToDest,
}

parentPort.on('message', async (data) => {
  runJob(data)
})

async function runJob({ id, job, params }) {
  const jobRun = jobs[job]
  const result = await jobRun(params)
  parentPort.postMessage({ id, result })
}
