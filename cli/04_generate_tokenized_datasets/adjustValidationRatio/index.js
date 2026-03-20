function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value))
}

function interpolate(value, inputStart, inputEnd, outputStart, outputEnd) {
  if (inputStart === inputEnd) return outputEnd
  const progress = (value - inputStart) / (inputEnd - inputStart)
  return outputStart + progress * (outputEnd - outputStart)
}

function interpolateLog(value, inputStart, inputEnd, outputStart, outputEnd) {
  const safeValue = Math.max(value, 1)
  const safeInputStart = Math.max(inputStart, 1)
  const safeInputEnd = Math.max(inputEnd, safeInputStart + 1)
  const progress =
    (Math.log10(safeValue) - Math.log10(safeInputStart)) /
    (Math.log10(safeInputEnd) - Math.log10(safeInputStart))
  return outputStart + clamp(progress, 0, 1) * (outputEnd - outputStart)
}

function buildPlan(sampleCount, range, targetValidationCount) {
  if (sampleCount <= 1) {
    return {
      range,
      validationCount: 0,
      validationRatio: 0,
    }
  }

  const validationCount = clamp(
    Math.round(targetValidationCount),
    1,
    sampleCount - 1
  )

  return {
    range,
    validationCount,
    validationRatio: validationCount / sampleCount,
  }
}

export function resolveValidationPlan(sampleCount) {
  if (sampleCount <= 200) {
    return buildPlan(
      sampleCount,
      'extra_small',
      Math.max(25, sampleCount * 0.25)
    )
  }

  if (sampleCount <= 2_000) {
    return buildPlan(
      sampleCount,
      'small',
      interpolate(sampleCount, 200, 2_000, 50, 200)
    )
  }

  if (sampleCount <= 20_000) {
    return buildPlan(
      sampleCount,
      'medium',
      interpolate(sampleCount, 2_000, 20_000, 200, 1_000)
    )
  }

  return buildPlan(
    sampleCount,
    'large',
    interpolateLog(sampleCount, 20_000, 1_000_000, 1_000, 5_000)
  )
}
