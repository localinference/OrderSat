import { getArgs } from './utils/getArgs/index.js'
import FastGlob from 'fast-glob'
import { Worker } from 'worker_threads'
import { tmpdir } from 'os'
import { join, basename, extname } from 'path'
import fs from 'fs/promises'
import { WASMagic } from 'wasmagic'
import { pdfToImage } from './worker/jobs/pdfToImage/index.js'
import { toUint8Array } from '@z-base/bytecodec'
import { writeUniqueToDest } from './worker/jobs/writeUniqueToDest/index.js'
import { cliui } from '@poppinss/cliui'
import { wait } from './utils/wait/index.js'
import os from 'os'
import { ensurePath } from './utils/ensurePath/index.js'
import { runWithWorker } from './runWithWorker/index.js'
const t0 = performance.now()
const ui = cliui()
/*************************************************/
const { languages, destinationPath } = getArgs()
/*************************************************/
const paths = {}
for (const language of languages) {
  console.log(`Looking up data sources for language: "${language}".\n`)
  await wait(1000)

  paths[language] = await FastGlob.async(`./models/.data/${language}/**/*.zip`)
  ui.logger.success(
    `Found possible sources:\n${(() => {
      let out = ''
      for (const path of paths[language]) {
        out += `"${path}"\n`
      }
      return out
    })()}
    \n\n\n`
  )
}
/***************************************************/
await wait(2500)
/***************************************************/
console.log(`Creating a temp dir for zip unpacking.\n`)
await wait(1000)

const tempRoot = await fs.mkdtemp(join(tmpdir(), '.data-unpack-'))
if (tempRoot) {
  ui.logger.success(`Created temp dir at "${tempRoot}".\n\n\n`)
} else throw new Error('Unable to make a temp dir.')
/****************/
await wait(2500)
/***************************************************/
for (const [language, routes] of Object.entries(paths)) {
  console.log(`Starting to unpack sources for language: "${language}".\n`)

  const running = []

  for (const route of routes) {
    const destinationPath = join(
      tempRoot,
      language,
      basename(route, extname(route))
    )
    try {
      console.log(`Unpacking "${route}".\n`)
      running.push(
        runWithWorker('unpackArchive', { path: route, dest: destinationPath })
      )
    } catch (err) {
      ui.logger.error(`Failed to unpack "${route}", because of "${err}".`)
    }
  }
  for (const run of running) {
    await run
  }
  ui.logger.success(
    `Successfully unpacked sources for language: "${language}".\n\n\n`
  )
}
/************/
await wait(2500)
/************/
for (const language of languages) {
  const files = await FastGlob.async('**/*', {
    cwd: join(tempRoot, language),
    dot: true,
    onlyFiles: true,
    markDirectories: false,
  })

  console.log(
    `Starting to filter ${files.length} files, to normalized unique input samples for language: "${language}".\n`
  )

  const availableParallelism = os.availableParallelism()
  const workerPool = []
  for (let i = 0; i < availableParallelism; i++) {
    const worker = new Worker(
      new URL('./workers/unpack.worker.js', import.meta.url),
      {
        workerData: { language },
      }
    )
    workerPool.push(worker)
  }
  await wait(2500)
  console.log(
    `Using up to ${availableParallelism} threads for image to text extraction.\n`
  )

  const magic = await WASMagic.create()
  for (const fileIndex in files) {
    const file = files[fileIndex]
    const content = await fs.readFile(`${tempRoot}/${file}`, {
      encoding: 'base64url',
    })
    const buffer = toUint8Array(Buffer.from(content, 'base64url'))
    try {
      const mime = magic.detect(buffer)
      if (mime) {
        if (mime.includes('image')) {
          new Promise((res, rej) => {
            const jobId = crypto.randomUUID()
            const workerIndex = fileIndex % (availableParallelism - 1)
            const selctedWorker = workerPool[workerIndex].postMessage({
              id: jobId,
              content: file,
            })
            selctedWorker.on('message', (ev) => {})
          })
          writeUniqueToDest(buffer, language, destinationPath)
        }
        if (mime.includes('pdf')) {
          const images = await pdfToImage(buffer)
          /** use image pool, and extract and then */
          writeUniqueToDest(buffer, language, destinationPath)
        }
        if (
          mime.includes('text') ||
          (mime.includes('application') && !mime.includes('pdf'))
        ) {
          writeUniqueToDest(buffer, language, destinationPath)
        }
      }
    } catch (err) {
      console.log(`Couldn't detect mime, because of "${err}"`)
    }
  }
  for (const worker of workerPool) {
    worker.terminate()
  }
}

/***************************************************/
console.log(
  `Extracted ${files.length} unique input samples to ${destinationPath}, in ${(performance.now() - t0) / 1000} seconds.`
)
await fs.rm(tempRoot, { recursive: true, force: true })
/***************************************************/
