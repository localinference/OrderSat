export function getInputPathFromOutputPath(outputPath) {
  return outputPath
    .replace('/outputs/', '/inputs/')
    .replace(/\.jsonld$/, '.txt')
}
