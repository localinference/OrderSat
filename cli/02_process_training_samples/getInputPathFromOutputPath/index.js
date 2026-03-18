export function getInputPathFromOutputPath(outputPath) {
  return outputPath
    .replace(/[\\/]outputs[\\/]/, (match) => match.replace('outputs', 'inputs'))
    .replace(/\.jsonld$/, '.txt')
}
