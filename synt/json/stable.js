export function stableJsonValue(value) {
  if (Array.isArray(value)) {
    return value.map((item) =>
      item === undefined ? null : stableJsonValue(item)
    )
  }

  if (value && typeof value === 'object') {
    const normalized = {}
    const keys = Object.keys(value).sort((left, right) =>
      left.localeCompare(right)
    )

    for (const key of keys) {
      const item = value[key]
      if (item === undefined) {
        continue
      }
      normalized[key] = stableJsonValue(item)
    }

    return normalized
  }

  return value
}

export function stableStringify(value) {
  return JSON.stringify(stableJsonValue(value))
}

export function stablePrettyStringify(value) {
  return JSON.stringify(stableJsonValue(value), null, 2)
}
