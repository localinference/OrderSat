function percentile(sortedNumbers, fraction) {
  if (sortedNumbers.length === 0) return 0
  const index = Math.min(
    sortedNumbers.length - 1,
    Math.max(0, Math.ceil(sortedNumbers.length * fraction) - 1)
  )
  return sortedNumbers[index]
}

export function summarizeLengths(values) {
  const sorted = [...values].sort((a, b) => a - b)
  const total = sorted.reduce((sum, value) => sum + value, 0)

  return {
    count: sorted.length,
    min: sorted[0] ?? 0,
    max: sorted[sorted.length - 1] ?? 0,
    avg: sorted.length === 0 ? 0 : total / sorted.length,
    p50: percentile(sorted, 0.5),
    p95: percentile(sorted, 0.95),
  }
}
