export function createRng(seedInput) {
  const seed = hashSeed(String(seedInput))
  let state = seed >>> 0

  return {
    float,
    int,
    pick,
    chance,
    shuffle,
    weightedPick,
    id,
  }

  function float() {
    state += 0x6d2b79f5
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }

  function int(min, max) {
    return Math.floor(float() * (max - min + 1)) + min
  }

  function pick(items) {
    if (!items.length) {
      throw new Error('Cannot pick from an empty array.')
    }
    return items[int(0, items.length - 1)]
  }

  function chance(probability) {
    return float() < probability
  }

  function shuffle(items) {
    const clone = [...items]
    for (let index = clone.length - 1; index > 0; index -= 1) {
      const swapIndex = int(0, index)
      const current = clone[index]
      clone[index] = clone[swapIndex]
      clone[swapIndex] = current
    }
    return clone
  }

  function weightedPick(entries) {
    const total = entries.reduce((sum, entry) => sum + entry.weight, 0)
    let cursor = float() * total

    for (const entry of entries) {
      cursor -= entry.weight
      if (cursor <= 0) {
        return entry.value
      }
    }

    return entries[entries.length - 1]?.value
  }

  function id(length, alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') {
    let out = ''
    for (let index = 0; index < length; index += 1) {
      out += alphabet[int(0, alphabet.length - 1)]
    }
    return out
  }
}

function hashSeed(value) {
  let hash = 2166136261
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }
  return hash >>> 0
}
