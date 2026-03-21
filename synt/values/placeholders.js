export function expandTemplatedValue(template, rng) {
  return template.replaceAll(/\[([^\]]+)\]/g, (_match, key) => {
    if (key === 'homeNumber') {
      return String(rng.int(1, 999))
    }
    if (key === 'floorNumber') {
      return String(rng.int(1, 20))
    }
    if (key === 'unitNumber') {
      return `${rng.int(1, 40)}${rng.pick(['A', 'B', 'C', 'D'])}`
    }
    if (key === 'poBoxNumber') {
      return String(rng.int(10, 9999))
    }
    if (key === 'postalCode') {
      return generatePostalCode(template, rng)
    }
    return key
  })
}

export function splitAddress(addressText) {
  const parts = addressText.split(',').map((part) => part.trim())
  return {
    fullText: addressText,
    streetAddress: parts[0] ?? '',
    addressLocality: parts[1] ?? '',
    addressRegion: parts[2] ?? '',
    postalCode: parts[3] ?? '',
    addressCountry: parts[4] ?? '',
  }
}

function generatePostalCode(template, rng) {
  if (template.includes('United Kingdom')) {
    return `${letters(rng, 2)}${rng.int(1, 9)} ${rng.int(1, 9)}${letters(rng, 2)}`
  }
  if (template.includes('Ireland')) {
    return `${letters(rng, 1)}${rng.int(10, 99)} ${letters(rng, 1)}${rng.int(100, 999)}`
  }
  if (template.includes('Canada')) {
    return `${letters(rng, 1)}${rng.int(0, 9)}${letters(rng, 1)} ${rng.int(0, 9)}${letters(rng, 1)}${rng.int(0, 9)}`
  }
  if (
    template.includes('Australia') ||
    template.includes('New Zealand') ||
    template.includes('Singapore')
  ) {
    return String(rng.int(1000, 9999))
  }
  return String(rng.int(10000, 99999))
}

function letters(rng, count) {
  let out = ''
  for (let index = 0; index < count; index += 1) {
    out += rng.pick('ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split(''))
  }
  return out
}
