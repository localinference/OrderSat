import { createWorker } from 'tesseract.js'
const tesseractWorkers = new Map()

export async function getTextFromImage({ image, language }) {
  try {
    let worker = tesseractWorkers.get(language)

    if (!worker) {
      console.log(
        `[OCR] Creating tesseract worker for language "${language}".\n`
      )
      worker = await createWorker(language)
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
