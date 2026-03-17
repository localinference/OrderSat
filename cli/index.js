import { getArgs } from './utils/getArgs/index.js'
import FastGlob from 'fast-glob'
import { tmpdir } from 'os'
import { join, extname } from 'path'
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
const machineLearningRoot = './machine_learning'
const dataSourcesRoot = `${machineLearningRoot}/data_sources`
const outputRoot = `${machineLearningRoot}/training_samples/inputs`
const maxConcurrentFiles = Math.max(availableParallelism * 2, 1)
const supportedArchiveExtensions = [
  '.nar',
  '.zip',
  '.tar',
  '.tgz',
  '.tar.gz',
  '.gz',
]

try {
  /*************************************************/
  const { languages } = getArgs()
  const outputGlobRoot = outputRoot.replaceAll('\\', '/')
  /*************************************************/
  await assertRequiredDirectory(machineLearningRoot)
  await assertRequiredDirectory(dataSourcesRoot)
  console.log(
    `[CLI] Starting input sample preparation.\nLanguages: ${languages.join(', ')}.\nDestination: "${outputRoot}".\nFile concurrency per language: ${maxConcurrentFiles}.\n\n`
  )
  console.log(
    `[CLI] Supported archive extensions: ${supportedArchiveExtensions.join(', ')}.\n`
  )
  const paths = Object.fromEntries(
    await Promise.all(
      languages.map(async (language) => {
        const languageSourceRoot = getLanguageSourceRoot(language)

        await assertRequiredDirectory(languageSourceRoot)

        console.log(
          `[CLI] Looking up data sources for language: "${language}" under "${languageSourceRoot}".\n`
        )

        const languagePaths = [
          ...new Set(
            await FastGlob.async(getArchiveSourceGlobs(language), {
              onlyFiles: true,
            })
          ),
        ]

        if (languagePaths.length === 0) {
          throw new Error(
            `[CLI] No supported archives were found for language "${language}" under "${languageSourceRoot}".`
          )
        }

        console.log(
          `[CLI] Found ${languagePaths.length} possible sources for language "${language}":\n${(() => {
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
  console.log(
    `[CLI] Finished source discovery for ${languages.length} language(s).\n`
  )
  console.log(`[CLI] Creating a temp dir for archive unpacking.\n`)

  tempRoot = await fs.mkdtemp(join(tmpdir(), '.data-unpack-'))
  if (tempRoot) {
    console.log(`[CLI] Created temp dir at "${tempRoot}".\n\n\n`)
  } else throw new Error('Unable to make a temp dir.')

  await Promise.all(
    languages.map(async (language) => {
      const l0 = performance.now()
      const routes = paths[language]
      const magic = await WASMagic.create()
      console.log(
        `[CLI] MIME detector ready for language "${language}". Starting archive pipeline for ${routes.length} archive(s).\n`
      )

      await Promise.all(
        routes.map(async (route) => {
          const unpackDestinationPath = getArchiveTempPath(
            tempRoot,
            language,
            route
          )

          console.log(
            `[CLI] Queueing archive unpack for "${route}" into "${unpackDestinationPath}".\n`
          )
          await runWithWorker('unpackArchive', {
            path: route,
            dest: unpackDestinationPath,
          })

          const files = await FastGlob.async('**/*', {
            cwd: unpackDestinationPath,
            dot: true,
            onlyFiles: true,
          })

          console.log(
            `[CLI] Archive "${route}" unpacked into ${files.length} file(s). Starting processing immediately for language "${language}".\n`
          )

          await runWithConcurrency(files, maxConcurrentFiles, async (file) => {
            await processExtractedFile({
              file,
              filePath: join(unpackDestinationPath, file),
              language,
              magic,
            })
          })

          console.log(
            `[CLI] Finished processing archive "${route}" for language "${language}".\n`
          )
        })
      )

      const uniquesLength = await getGlobLength(
        `${outputGlobRoot}/${language}/*.txt`
      )
      console.log(
        `[CLI] Created ${uniquesLength} unique "${language}" input sample(s) in ${Math.round((performance.now() - l0) / 1000 / 60)} minutes.\n\n\n`
      )
    })
  )

  /***************************************************/
  console.log(
    `[CLI] Prepared all input samples in ${Math.round((performance.now() - t0) / 1000 / 60)} minutes.\n`
  )
  /***************************************************/
} catch (err) {
  ui.logger.error(err?.stack ?? String(err))
  process.exitCode = 1
} finally {
  try {
    console.log(`[CLI] Closing worker pool.\n`)
    await closeWorkerPool()
    console.log(`[CLI] Worker pool closed.\n`)
  } catch (err) {
    ui.logger.error(`Failed to close worker pool, because of "${err}".`)
    process.exitCode = 1
  }

  if (tempRoot) {
    try {
      console.log(`[CLI] Cleaning temp dir "${tempRoot}".\n`)
      await fs.rm(tempRoot, { recursive: true, force: true })
      console.log(`[CLI] Cleaned temp dir "${tempRoot}".\n`)
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

function getArchiveTempPath(tempRoot, language, route) {
  const languagePrefix = `${getLanguageSourceRoot(language)}/`
  const relativeRoute = route.startsWith(languagePrefix)
    ? route.slice(languagePrefix.length)
    : route

  return join(
    tempRoot,
    language,
    relativeRoute.slice(0, -extname(relativeRoute).length)
  )
}

function getArchiveSourceGlobs(language) {
  return supportedArchiveExtensions.map(
    (extension) => `${getLanguageSourceRoot(language)}/**/*${extension}`
  )
}

function getLanguageSourceRoot(language) {
  return `${dataSourcesRoot}/${language}`
}

async function assertRequiredDirectory(path) {
  try {
    const stats = await fs.stat(path)

    if (!stats.isDirectory()) {
      throw new Error(`[CLI] Required directory "${path}" is not a directory.`)
    }
  } catch (err) {
    if (err?.code === 'ENOENT') {
      throw new Error(`[CLI] Required directory "${path}" does not exist.`)
    }

    throw err
  }
}

async function processExtractedFile({ file, filePath, language, magic }) {
  const buffer = await fs.readFile(filePath)

  try {
    const mime = magic.detect(buffer)
    if (!mime) {
      console.log(
        `[CLI] Skipping "${file}" for language "${language}" because no mime type was detected.\n`
      )
      return
    }

    console.log(
      `[CLI] Processing "${file}" for language "${language}" as "${mime}".\n`
    )

    if (mime.includes('image')) {
      console.log(
        `[CLI] Requesting OCR for image "${file}" in language "${language}".\n`
      )
      const textFromImageResult = await runWithWorker('getTextFromImage', {
        language,
        image: buffer,
      })
      await writeUniqueToDest(fromString(textFromImageResult), language)
      return
    }

    if (mime.includes('pdf')) {
      console.log(
        `[CLI] Rendering PDF "${file}" to images for language "${language}".\n`
      )
      const images = await runWithWorker('pdfToImages', {
        pdfBuffer: buffer,
      })
      console.log(
        `[CLI] PDF "${file}" produced ${images.length} image page(s); requesting OCR for language "${language}".\n`
      )
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

      await writeUniqueToDest(fromString(text), language)
      return
    }

    if (
      mime.includes('text') ||
      (mime.includes('application') && !mime.includes('pdf'))
    ) {
      console.log(
        `[CLI] Writing text-like file "${file}" directly for language "${language}".\n`
      )
      await writeUniqueToDest(buffer, language)
    }
  } catch (err) {
    console.log(
      `[CLI] Failed while processing "${file}" for language "${language}", because of "${err}".\n\n\n`
    )
  }
}
