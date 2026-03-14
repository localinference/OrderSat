import { getArgs } from './utils/getArgs/index.js'
import FastGlob from 'fast-glob'
import { Worker } from 'worker_threads'
import { tmpdir } from 'os'
import { join, basename, extname } from 'path'
import fs from 'fs/promises'
import { WASMagic } from 'wasmagic'

const t0 = performance.now()
/*************************************************/
const { languages } = getArgs()
console.log('languages =', languages)
/*************************************************/
const paths = {}
for (const language of languages) {
  paths[language] = await FastGlob.async(`./models/.data/${language}/**/*.zip`)
}
console.log('paths =', paths)
/***************************************************/
const tempRoot = await fs.mkdtemp(join(tmpdir(), '.data-unpack-'))
console.log('tempRoot =', tempRoot)
/***************************************************/
const runUnpackWorker = (route, destinationPath) =>
  new Promise((resolve, reject) => {
    const worker = new Worker(
      new URL('./workers/unpack.worker.js', import.meta.url),
      {
        workerData: { route, destinationPath },
      }
    )

    worker.once('message', resolve)
    worker.once('error', reject)
    worker.once('exit', (code) => {
      if (code !== 0) reject(new Error(`Worker exited with code ${code}.`))
    })
  })
/***************************************************/
const unpacks = {}
for (const [language, routes] of Object.entries(paths)) {
  unpacks[language] = []

  for (const route of routes) {
    const destinationPath = join(
      tempRoot,
      language,
      basename(route, extname(route))
    )

    await fs.mkdir(destinationPath, { recursive: true })

    try {
      console.log(`Unpacking ${route}.`)
      unpacks[language].push(await runUnpackWorker(route, destinationPath))
    } catch (err) {
      console.log(`Failed to unpack ${route}, because of "${err}".`)
    }
  }
}
console.log('unpacks =', unpacks)
/***************************************************/
const files = await FastGlob.async('**/*', {
  cwd: tempRoot,
  dot: true,
  onlyFiles: true,
  markDirectories: false,
})
console.log('files =', files)
/***************************************************/
const magic = await WASMagic.create()
for (const file of files) {
  const content = await fs.readFile(`${tempRoot}/${file}`, {
    encoding: 'base64url',
  })
  const buffer = Buffer.from(content, 'base64url')
  try {
    const mime = magic.detect(buffer)
    if (mime) {
      if (mime.includes('text')) {
        console.log(mime)
      }
    }
  } catch (err) {
    console.log(`Couldn't detect mime, because of "${err}"`)
  }
}
/***************************************************/
const listenToImageToTextWorker = (route, destinationPath) =>
  new Promise((resolve, reject) => {
    const worker = new Worker(
      new URL('./workers/unpack.worker.js', import.meta.url),
      {
        workerData: {},
      }
    )

    worker.once('message', resolve)
    worker.once('error', reject)
    worker.once('exit', (code) => {
      if (code !== 0) reject(new Error(`Worker exited with code ${code}.`))
    })
  })
/***************************************************/
console.log(
  `Extracted ${files.length} input samples in ${(performance.now() - t0) / 1000} seconds.`
)

await fs.rm(tempRoot, { recursive: true, force: true })
