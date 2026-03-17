import { getArgs } from './utils/getArgs/index.js'
import FastGlob from 'fast-glob'
import { tmpdir } from 'os'
import { join, basename, extname } from 'path'
import fs from 'fs/promises'
import { WASMagic } from 'wasmagic'
import { fromString, toUint8Array } from '@z-base/bytecodec'
import { writeUniqueToDest } from './writeUniqueToDest/index.js'
import { cliui } from '@poppinss/cliui'
import { wait } from './utils/wait/index.js'
import { closeWorkerPool, runWithWorker } from './runWithWorker/index.js'
import { getGlobLength } from './utils/getGlobLength/index.js'

const t0 = performance.now()
const ui = cliui()
let tempRoot

try {
  /*************************************************/
  const { languages, destinationPath } = getArgs()
  const outputGlobRoot = destinationPath.replaceAll('\\', '/')
  /*************************************************/
  const paths = {}
  for (const language of languages) {
    console.log(`Looking up data sources for language: "${language}".\n`)
    await wait(1000)

    paths[language] = await FastGlob.async(
      `./models/.data/${language}/**/*.zip`
    )
    console.log(
      `Found ${paths[language].length} possible sources:\n${(() => {
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

  tempRoot = await fs.mkdtemp(join(tmpdir(), '.data-unpack-'))
  if (tempRoot) {
    console.log(`Created temp dir at "${tempRoot}".\n\n\n`)
  } else throw new Error('Unable to make a temp dir.')
  /****************/
  await wait(2500)
  /***************************************************/
  for (const [language, routes] of Object.entries(paths)) {
    console.log(`Starting to unpack sources for language: "${language}".\n`)

    const running = []

    for (const route of routes) {
      const unpackDestinationPath = join(
        tempRoot,
        language,
        basename(route, extname(route))
      )

      console.log(`Unpacking "${route}".\n`)
      running.push(
        runWithWorker('unpackArchive', {
          path: route,
          dest: unpackDestinationPath,
        })
      )
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
    const l0 = performance.now()
    const files = await FastGlob.async('**/*', {
      cwd: join(tempRoot, language),
      dot: true,
      onlyFiles: true,
    })

    console.log(
      `Starting to filter ${files.length} files, to normalized unique input samples for language: "${language}".\n`
    )
    const magic = await WASMagic.create()
    for (const fileIndex in files) {
      const file = files[fileIndex]
      const content = await fs.readFile(join(tempRoot, language, file), {
        encoding: 'base64url',
      })
      const buffer = toUint8Array(Buffer.from(content, 'base64url'))
      try {
        const mime = magic.detect(buffer)
        if (mime) {
          if (mime.includes('image')) {
            const textFromImageResult = await runWithWorker(
              'getTextFromImage',
              {
                language: language,
                image: buffer,
              }
            )
            await writeUniqueToDest(
              fromString(textFromImageResult),
              language,
              destinationPath
            )
          }
          if (mime.includes('pdf')) {
            const running = []
            const images = await runWithWorker('pdfToImages', {
              pdfBuffer: buffer,
            })
            for (const image of images) {
              running.push(
                runWithWorker('getTextFromImage', {
                  language: language,
                  image: image.data,
                })
              )
            }
            let text = ''
            for (const run of running) {
              text += await run
            }
            await writeUniqueToDest(fromString(text), language, destinationPath)
          }
          if (
            mime.includes('text') ||
            (mime.includes('application') && !mime.includes('pdf'))
          ) {
            await writeUniqueToDest(buffer, language, destinationPath)
          }
        }
      } catch (err) {
        console.log(`Couldn't detect mime, because of "${err}"\n\n\n`)
      }
    }
    const uniquesLength = await getGlobLength(
      `${outputGlobRoot}/${language}/*.txt`
    )
    ui.logger.success(
      `Created ${uniquesLength} unique "${language}" input samples in ${(performance.now() - l0) / 1000} seconds.\n\n\n`
    )
  }

  /***************************************************/
  console.log(
    `Preapared all input samples in ${(performance.now() - t0) / 1000} seconds.`
  )
  /***************************************************/
} catch (err) {
  ui.logger.error(err?.stack ?? String(err))
  process.exitCode = 1
} finally {
  try {
    await closeWorkerPool()
  } catch (err) {
    ui.logger.error(`Failed to close worker pool, because of "${err}".`)
    process.exitCode = 1
  }

  if (tempRoot) {
    try {
      await fs.rm(tempRoot, { recursive: true, force: true })
    } catch (err) {
      ui.logger.error(
        `Failed to clean temp dir "${tempRoot}", because of "${err}".`
      )
      process.exitCode = 1
    }
  }
}
