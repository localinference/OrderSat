import { mkdir } from 'fs/promises'
import { resolve } from 'path'
import { createWorker, OEM } from 'tesseract.js'

const tesseractWorkers = new Map()
const tesseractCachePath = resolve(process.cwd(), 'src', '.cache', 'tesseract')
const tesseractCacheReady = mkdir(tesseractCachePath, { recursive: true })

export async function getTextFromImage({ image, language }) {
  try {
    let worker = tesseractWorkers.get(language)

    if (!worker) {
      await tesseractCacheReady
      console.log(
        `[OCR] Creating tesseract worker for language "${language}" using cache "${tesseractCachePath}".\n`
      )
      worker = await createWorker(language, OEM.LSTM_ONLY, {
        cachePath: tesseractCachePath,
      })
      tesseractWorkers.set(language, worker)
    }
    console.log(
      `[OCR] Starting OCR for language "${language}" on image data (${image.length} bytes).\n`
    )
    const result = await worker.recognize(image)
    console.log(
      `[OCR] Finished OCR for language "${language}". Extracted ${result.data.text.length} character(s).\n`
    )
    return result.data.text
  } catch (err) {
    throw err
  }
}
