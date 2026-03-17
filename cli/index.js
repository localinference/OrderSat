import { getArgs } from './utils/getArgs/index.js'
import FastGlob from 'fast-glob'
import { tmpdir } from 'os'
import { join, basename, extname } from 'path'
import fs from 'fs/promises'
import { WASMagic } from 'wasmagic'
import { fromString } from '@z-base/bytecodec'
import { writeUniqueToDest } from './writeUniqueToDest/index.js'
import { cliui } from '@poppinss/cliui'
import {
  availableParallelism,
  closeWorkerPool,
  runWithWorker,
} from './runWithWorker/index.js'
import { getGlobLength } from './utils/getGlobLength/index.js'

const t0 = performance.now()
const ui = cliui()
let tempRoot
const maxConcurrentFiles = Math.max(availableParallelism * 2, 1)

try {
  /*************************************************/
  const { languages, destinationPath } = getArgs()
  const outputGlobRoot = destinationPath.replaceAll('\\', '/')
  /*************************************************/
  const paths = Object.fromEntries(
    await Promise.all(
      languages.map(async (language) => {
        console.log(`Looking up data sources for language: "${language}".\n`)

        const languagePaths = await FastGlob.async(
          `./models/.data/${language}/**/*.zip`
        )

        console.log(
          `Found ${languagePaths.length} possible sources:\n${(() => {
            let out = ''
            for (const path of languagePaths) {
              out += `"${path}"\n`
            }
            return out
          })()}
          \n\n\n`
        )

        return [language, languagePaths]
      })
    )
  )
  console.log(`Creating a temp dir for zip unpacking.\n`)

  tempRoot = await fs.mkdtemp(join(tmpdir(), '.data-unpack-'))
  if (tempRoot) {
    console.log(`Created temp dir at "${tempRoot}".\n\n\n`)
  } else throw new Error('Unable to make a temp dir.')

  await Promise.all(
    Object.entries(paths).map(async ([language, routes]) => {
      console.log(`Starting to unpack sources for language: "${language}".\n`)

      await Promise.all(
        routes.map((route) => {
          const unpackDestinationPath = join(
            tempRoot,
            language,
            basename(route, extname(route))
          )

          console.log(`Unpacking "${route}".\n`)
          return runWithWorker('unpackArchive', {
            path: route,
            dest: unpackDestinationPath,
          })
        })
      )

      ui.logger.success(
        `Successfully unpacked sources for language: "${language}".\n\n\n`
      )
    })
  )

  await Promise.all(
    languages.map(async (language) => {
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

      await runWithConcurrency(files, maxConcurrentFiles, async (file) => {
        const buffer = await fs.readFile(join(tempRoot, language, file))

        try {
          const mime = magic.detect(buffer)
          if (!mime) {
            return
          }

          if (mime.includes('image')) {
            const textFromImageResult = await runWithWorker(
              'getTextFromImage',
              {
                language,
                image: buffer,
              }
            )
            await writeUniqueToDest(
              fromString(textFromImageResult),
              language,
              destinationPath
            )
            return
          }

          if (mime.includes('pdf')) {
            const images = await runWithWorker('pdfToImages', {
              pdfBuffer: buffer,
            })
            const text = (
              await Promise.all(
                images.map((image) =>
                  runWithWorker('getTextFromImage', {
                    language,
                    image: image.data,
                  })
                )
              )
            ).join('')

            await writeUniqueToDest(fromString(text), language, destinationPath)
            return
          }

          if (
            mime.includes('text') ||
            (mime.includes('application') && !mime.includes('pdf'))
          ) {
            await writeUniqueToDest(buffer, language, destinationPath)
          }
        } catch (err) {
          console.log(`Couldn't detect mime, because of "${err}"\n\n\n`)
        }
      })

      const uniquesLength = await getGlobLength(
        `${outputGlobRoot}/${language}/*.txt`
      )
      ui.logger.success(
        `Created ${uniquesLength} unique "${language}" input samples in ${(performance.now() - l0) / 1000} seconds.\n\n\n`
      )
    })
  )

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

async function runWithConcurrency(items, concurrency, run) {
  let index = 0
  const workerCount = Math.min(items.length, concurrency)

  await Promise.all(
    Array.from({ length: workerCount }, async () => {
      for (;;) {
        const currentIndex = index
        index += 1

        if (currentIndex >= items.length) {
          return
        }

        await run(items[currentIndex], currentIndex)
      }
    })
  )
}
