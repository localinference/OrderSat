export function chooseCustomerName(values, rng) {
  return choosePersonName(values, rng)
}

export function chooseBrokerName(values, rng) {
  return choosePersonName(values, rng)
}

export function chooseLocale(values, rng) {
  return rng.pick(values.atoms.geography.locales)
}

export function chooseMerchantName(values, rng) {
  return interpolatePattern(
    rng.pick(values.patterns.companyNameFormats),
    {
      prefix: rng.pick(values.atoms.companies.prefixes),
      root: rng.pick(values.atoms.companies.roots),
      suffix: rng.pick(values.atoms.companies.suffixes),
      family: rng.pick(values.atoms.people.lastNames),
      family2: rng.pick(values.atoms.people.lastNames),
    }
  )
}

export function chooseLogisticsProviderName(values, rng) {
  return interpolatePattern(
    rng.pick(values.patterns.logisticsCompanyFormats),
    {
      prefix: rng.pick(values.atoms.companies.logisticsPrefixes),
      root: rng.pick(values.atoms.companies.logisticsRoots),
      suffix: rng.pick(values.atoms.companies.logisticsSuffixes),
    }
  )
}

export function chooseAddress(locale, rng, values) {
  const city = rng.pick(locale.cities)
  const streetName = rng.pick(city.streets)
  const houseNumber = String(rng.int(1, 999))
  const secondLine = rng.chance(0.28)
    ? `${rng.pick(values.atoms.companies.suiteLabels)} ${rng.int(1, 40)}`
    : null
  const streetAddress = [houseNumber, streetName, secondLine]
    .filter(Boolean)
    .join(', ')
  const postalCode = formatMask(rng.pick(locale.postalFormats), rng)
  const fullText = [
    streetAddress,
    city.name,
    city.region,
    postalCode,
    locale.countryName,
  ].join(', ')

  return {
    fullText,
    streetAddress,
    addressLocality: city.name,
    addressRegion: city.region,
    postalCode,
    addressCountry: locale.countryName,
    addressCountryCode: locale.countryCode,
    currency: locale.currency,
    locale,
  }
}

export function chooseItems(values, rng, blueprint) {
  const quantityRange =
    blueprint === 'service-confirmation'
      ? values.atoms.commerce.ranges.serviceQuantity
      : values.atoms.commerce.ranges.productQuantity
  const itemCount =
    blueprint === 'service-confirmation'
      ? rng.int(1, Math.min(3, quantityRange.max))
      : rng.int(1, Math.min(5, quantityRange.max))

  const categoryNames = values.patterns.blueprintCatalogs[blueprint]
  const candidates = uniqueByName(
    categoryNames.flatMap((categoryName) =>
      buildCategoryCandidates(values, rng, categoryName, itemCount * 3)
    )
  )

  return rng.shuffle(candidates).slice(0, itemCount).map((candidate) => {
    const categoryConfig = values.catalogs.composedOffers[candidate.categoryName]
    const type = inferOfferType(candidate.name, blueprint)
    const effectiveQuantityRange =
      type === 'Service'
        ? values.atoms.commerce.ranges.serviceQuantity
        : categoryConfig?.quantityRange ?? values.atoms.commerce.ranges.productQuantity
    const effectivePriceRange = categoryConfig?.priceRange ?? {
      min: type === 'Service' ? 18 : 4,
      max: type === 'Service' ? 180 : 120,
    }
    const quantity = rng.int(
      effectiveQuantityRange.min,
      effectiveQuantityRange.max
    )
    const unitPrice = randomMoney(
      rng,
      effectivePriceRange.min,
      effectivePriceRange.max
    )
    const linePrice = roundMoney(quantity * unitPrice)

    return {
      type,
      name: candidate.name,
      sku: formatMask(rng.pick(values.patterns.skuFormats), rng),
      quantity,
      unitText: 'item',
      unitPrice,
      linePrice,
    }
  })
}

export function chooseCurrency(locale) {
  return locale.currency
}

export function choosePayment(values, rng) {
  const method = rng.pick(values.atoms.commerce.paymentMethods)
  const status = rng.weightedPick([
    { value: values.atoms.commerce.paymentStatuses[0], weight: 5 },
    { value: values.atoms.commerce.paymentStatuses[1], weight: 2 },
    { value: values.atoms.commerce.paymentStatuses[2], weight: 1 },
    { value: values.atoms.commerce.paymentStatuses[3], weight: 1 },
  ])

  return {
    ...method,
    paymentStatusDisplay: status.display,
    paymentStatusUrl: status.schemaUrl,
  }
}

export function chooseOrderStatus(values, rng, blueprint) {
  if (blueprint === 'shipping-notice') {
    return rng.weightedPick([
      { value: values.atoms.commerce.orderStatuses.shipping[0], weight: 3 },
      { value: values.atoms.commerce.orderStatuses.shipping[1], weight: 2 },
      { value: values.atoms.commerce.orderStatuses.shipping[2], weight: 1 },
    ])
  }

  if (blueprint === 'service-confirmation') {
    return rng.weightedPick([
      { value: values.atoms.commerce.orderStatuses.service[0], weight: 3 },
      { value: values.atoms.commerce.orderStatuses.service[1], weight: 1 },
    ])
  }

  return rng.weightedPick([
    { value: values.atoms.commerce.orderStatuses.default[0], weight: 4 },
    { value: values.atoms.commerce.orderStatuses.default[1], weight: 2 },
    { value: values.atoms.commerce.orderStatuses.default[2], weight: 1 },
  ])
}

export function chooseDiscountPercent(values, rng) {
  return randomPercent(rng, values.atoms.commerce.ranges.discountPercent)
}

export function chooseTaxRate(values, rng, taxType = 'default') {
  const percent = randomPercent(
    rng,
    values.atoms.commerce.ranges.taxPercent[taxType]
  )
  return percent / 100
}

export function generateOrderNumber(values, rng) {
  return formatMask(rng.pick(values.patterns.orderNumberFormats), rng)
}

export function generateTrackingNumber(values, rng) {
  return formatMask(rng.pick(values.patterns.trackingNumberFormats), rng)
}

export function generateTelephone(locale, rng) {
  return formatMask(rng.pick(locale.phoneFormats), rng)
}

export function generateEmail(values, name, merchantName, rng) {
  const parts = name.split(/\s+/).filter(Boolean)
  const tokens = {
    first: slugify(parts[0] ?? 'user').replaceAll('-', '.'),
    last: slugify(parts.at(-1) ?? 'user').replaceAll('-', '.'),
    merchant: slugify(merchantName),
    nn: String(rng.int(1, 99)),
  }
  const local = interpolatePattern(
    rng.pick(values.patterns.emailLocalFormats),
    tokens
  )
  const domain = rng.pick(values.atoms.commerce.emailDomains)
  return `${local}@${domain}`
}

export function generateTaxId(locale, rng) {
  return formatMask(rng.pick(locale.taxIdFormats), rng)
}

export function choosePromoCode(values, rng) {
  const percent = String(
    randomPercent(rng, values.atoms.commerce.ranges.promoPercent)
  )
  const tokens = {
    prefix: rng.pick(values.atoms.commerce.promoPrefixes),
    theme: rng.pick(values.atoms.commerce.promoThemes),
    percent,
    year2: String(24 + rng.int(0, 6)),
  }
  const raw = interpolatePattern(
    rng.pick(values.patterns.promoCodeFormats),
    tokens
  )
  return applyCaseStyle(raw, rng)
}

export function chooseServiceMode(values, rng) {
  return rng.pick(values.atoms.commerce.serviceModes)
}

export function chooseReceiptType(values, rng) {
  return rng.pick(values.atoms.commerce.receiptTypes)
}

export function generateCustomerId(values, rng) {
  return formatMask(rng.pick(values.patterns.customerIdFormats), rng)
}

export function generateRegisterNumber(values, rng) {
  return formatMask(rng.pick(values.patterns.registerNumberFormats), rng)
}

export function generateWebsiteUrl(values, merchantName, rng) {
  const host = interpolatePattern(
    rng.pick(values.patterns.websiteHostFormats),
    {
      merchant: slugify(merchantName),
    }
  )
  const suffix = rng.pick(values.atoms.companies.domains)
  return `https://${host}${suffix}`
}

export function randomMoney(rng, min, max) {
  return Number((rng.int(min * 100, max * 100) / 100).toFixed(2))
}

export function randomPercent(rng, range) {
  const decimals = range.decimals ?? 0
  const factor = 10 ** decimals
  return Number(
    (
      rng.int(
        Math.round(range.min * factor),
        Math.round(range.max * factor)
      ) / factor
    ).toFixed(decimals)
  )
}

export function toMoneyString(value) {
  return value.toFixed(2).replace(/\.00$/, '').replace(/(\.\d)0$/, '$1')
}

export function slugify(value) {
  return value
    .toLowerCase()
    .replaceAll(/[^a-z0-9]+/g, '-')
    .replaceAll(/^-+|-+$/g, '')
}

function choosePersonName(values, rng) {
  const first = rng.pick(values.atoms.people.firstNames)
  const last = rng.pick(values.atoms.people.lastNames)
  const middle = rng.pick(values.atoms.people.middleInitials)
  return interpolatePattern(
    rng.pick(values.patterns.personNameFormats),
    {
      first,
      middle,
      last,
    }
  )
}

function buildCategoryCandidates(values, rng, categoryName, generatedCount) {
  const fixed = (values.catalogs.offers[categoryName] ?? []).map((name) => ({
    name,
    categoryName,
  }))
  const composed = Array.from({ length: generatedCount }, () => ({
    name: generateOfferName(values.catalogs.composedOffers[categoryName], rng),
    categoryName,
  })).filter((entry) => entry.name)

  return [...fixed, ...composed]
}

function generateOfferName(categoryConfig, rng) {
  if (!categoryConfig) {
    return null
  }

  return interpolatePattern(rng.pick(categoryConfig.formats), {
    adj: rng.pick(categoryConfig.adjectives),
    noun: rng.pick(categoryConfig.nouns),
    qual: rng.pick(categoryConfig.qualifiers),
  }).replaceAll(/\s+/g, ' ').trim()
}

function uniqueByName(entries) {
  const seen = new Set()
  const out = []

  for (const entry of entries) {
    if (!entry?.name || seen.has(entry.name)) {
      continue
    }
    seen.add(entry.name)
    out.push(entry)
  }

  return out
}

function interpolatePattern(pattern, tokens) {
  return pattern.replaceAll(/\{([^}]+)\}/g, (_match, key) => tokens[key] ?? key)
}

function applyCaseStyle(value, rng) {
  const style = rng.pick(['upper', 'lower', 'title'])
  if (style === 'upper') {
    return value.toUpperCase()
  }
  if (style === 'lower') {
    return value.toLowerCase()
  }
  return value
    .split(/([_-])/)
    .map((part) =>
      part === '_' || part === '-'
        ? part
        : part.charAt(0).toUpperCase() + part.slice(1).toLowerCase()
    )
    .join('')
}

function formatMask(mask, rng) {
  let out = ''

  for (const character of mask) {
    if (character === 'A') {
      out += rng.pick('ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split(''))
      continue
    }
    if (character === 'a') {
      out += rng.pick('abcdefghijklmnopqrstuvwxyz'.split(''))
      continue
    }
    if (character === '#') {
      out += String(rng.int(0, 9))
      continue
    }
    if (character === '*') {
      out += rng.pick('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'.split(''))
      continue
    }
    out += character
  }

  return out
}

function inferOfferType(name, blueprint) {
  if (blueprint === 'service-confirmation') {
    return 'Service'
  }

  return /service|session|subscription|ticket|pass|booking|consultation|lesson|plan|access|inspection|support|repair/i.test(
    name
  )
    ? 'Service'
    : 'Product'
}

function roundMoney(value) {
  return Number(value.toFixed(2))
}
