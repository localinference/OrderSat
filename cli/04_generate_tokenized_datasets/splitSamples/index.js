function resolveValidationCount(total, options) {
  if (total <= 1) return 0

  if (options.validationCount !== null) {
    return Math.min(total - 1, options.validationCount)
  }

  if (options.validationRatio === null) {
    throw new Error('validationRatio or validationCount must be provided')
  }

  const derived = Math.round(total * options.validationRatio)
  return Math.min(total - 1, Math.max(1, derived))
}

export function splitSamples(samples, options) {
  const ordered = [...samples].sort((a, b) =>
    a.sample_id.localeCompare(b.sample_id)
  )
  const validationCount = resolveValidationCount(ordered.length, options)
  const validation = ordered.slice(0, validationCount)
  const train = ordered.slice(validationCount)

  return { train, validation }
}
