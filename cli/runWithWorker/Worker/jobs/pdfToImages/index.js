import { PDFParse } from 'pdf-parse'

const standardFontDataUrl = new URL(
  '../../../../../node_modules/pdf-parse/node_modules/pdfjs-dist/standard_fonts/',
  import.meta.url
).href

export async function pdfToImages({ pdfBuffer }) {
  const parser = new PDFParse({ data: pdfBuffer, standardFontDataUrl })
  try {
    console.log(
      `[PDF] Rendering PDF buffer (${pdfBuffer.length} bytes) to images.\n`
    )
    const result = await parser.getScreenshot({ scale: 1.5 })
    console.log(
      `[PDF] Finished rendering PDF buffer into ${result.pages.length} page image(s).\n`
    )
    return result.pages
  } finally {
    await parser.destroy()
  }
}
