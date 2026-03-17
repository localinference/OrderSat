import { createWorker } from 'tesseract.js'
const tesseractWorkers = new Map()

export async function getTextFromImage({ image, language }) {
  try {
    let worker = tesseractWorkers.get(language)

    if (!worker) {
      worker = await createWorker(language)
      tesseractWorkers.set(language, worker)
    }
    const result = await worker.recognize(image)
    return result.data.text
  } catch (err) {
    throw err
  }
}
