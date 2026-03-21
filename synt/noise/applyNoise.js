export function applyNoise(text, rng, profile) {
  if (profile === 'clean') {
    return text
  }

  let noisy = text
  const passes = profile === 'ocr-medium' ? 10 : 4

  for (let index = 0; index < passes; index += 1) {
    noisy = applySubstitution(noisy, rng)
    noisy = maybeWarpWhitespace(noisy, rng, profile)
  }

  return noisy
}

function applySubstitution(text, rng) {
  const substitutions = [
    ['0', 'O'],
    ['O', '0'],
    ['1', 'I'],
    ['I', '1'],
    ['l', '1'],
    ['S', '5'],
    ['5', 'S'],
    ['B', '8'],
    ['rn', 'm'],
    ['m', 'rn'],
    ['.', ','],
    [':', ';'],
  ]

  const [source, target] = rng.pick(substitutions)
  const indexes = findIndexes(text, source)
  if (!indexes.length) {
    return text
  }

  const start = rng.pick(indexes)
  return text.slice(0, start) + target + text.slice(start + source.length)
}

function maybeWarpWhitespace(text, rng, profile) {
  if (!rng.chance(profile === 'ocr-medium' ? 0.35 : 0.15)) {
    return text
  }

  const whitespaceRuns = [...text.matchAll(/\s+/g)]
  if (!whitespaceRuns.length) {
    return text
  }

  const match = rng.pick(whitespaceRuns)
  const replacement = rng.chance(0.5) ? ' ' : '\n'
  return (
    text.slice(0, match.index) +
    replacement +
    text.slice(match.index + match[0].length)
  )
}

function findIndexes(text, query) {
  const indexes = []
  let cursor = text.indexOf(query)
  while (cursor !== -1) {
    indexes.push(cursor)
    cursor = text.indexOf(query, cursor + 1)
  }
  return indexes
}
