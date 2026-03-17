import { PDFParse } from 'pdf-parse'

export async function pdfToImage(pdfBuffer) {
  const parser = new PDFParse(pdfBuffer)
  try {
    const result = await parser.getScreenshot({ scale: 1.5 })
    return result.pages
  } finally {
    await parser.destroy()
  }
}
