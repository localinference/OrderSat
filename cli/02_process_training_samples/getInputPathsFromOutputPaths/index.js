export function getInputPathsFromOutputPaths(outputPaths) {
  const inputPaths = outputPaths.map((path) =>
    path.replace('/outputs/', '/inputs/').replace(/\.jsonld$/, '.txt')
  )

  return inputPaths
}
