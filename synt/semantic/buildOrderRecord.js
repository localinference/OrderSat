import {
  chooseAddress,
  chooseBrokerName,
  chooseCurrency,
  chooseCustomerName,
  chooseItems,
  chooseMerchantName,
  chooseOrderStatus,
  choosePayment,
  choosePromoCode,
  chooseReceiptType,
  chooseServiceMode,
  generateEmail,
  generateOrderNumber,
  generateTaxId,
  generateTelephone,
  generateTrackingNumber,
  randomMoney,
  slugify,
} from '../values/factories.js'

export const renderers = [
  'plain-receipt',
  'email-summary',
  'html-order',
  'json-dump',
  'xml-summary',
  'csv-export',
]

const blueprints = [
  'retail-receipt',
  'online-confirmation',
  'shipping-notice',
  'service-confirmation',
]

const noiseProfiles = ['clean', 'ocr-light', 'ocr-medium']
const errorProfiles = ['none', 'uncertain', 'fatal', 'mixed']

export function getCoveragePlan(index) {
  const renderer = renderers[index % renderers.length]
  const rendererIndex = Math.floor(index / renderers.length)
  const blueprint = blueprints[rendererIndex % blueprints.length]
  const blueprintIndex = Math.floor(rendererIndex / blueprints.length)
  const noise = noiseProfiles[blueprintIndex % noiseProfiles.length]
  const noiseIndex = Math.floor(blueprintIndex / noiseProfiles.length)
  const errorProfile = errorProfiles[noiseIndex % errorProfiles.length]

  return {
    renderer,
    blueprint,
    noise,
    errorProfile,
    coverageKey: `${blueprint}:${renderer}:${noise}:${errorProfile}`,
  }
}

export function buildOrderRecord({ languageConfig, rng, plan, index }) {
  const merchantAddress = chooseAddress(languageConfig.values, rng)
  const customerAddress = chooseAddress(languageConfig.values, rng)
  const sellerName = chooseMerchantName(rng)
  const customerName = chooseCustomerName(languageConfig.values, rng)
  const currency = chooseCurrency(merchantAddress.addressCountry, rng)
  const items = chooseItems(languageConfig.values, rng, plan.blueprint)
  const subtotal = roundMoney(
    items.reduce((sum, item) => sum + item.linePrice, 0)
  )
  const discountAmount = rng.chance(0.3)
    ? roundMoney(subtotal * rng.pick([0.05, 0.1, 0.15]))
    : 0
  const taxableAmount = subtotal - discountAmount
  const taxRate =
    plan.blueprint === 'service-confirmation'
      ? rng.pick([0, 0.1, 0.2])
      : rng.pick([0, 0.05, 0.08, 0.2])
  const taxAmount = roundMoney(taxableAmount * taxRate)
  const total = roundMoney(taxableAmount + taxAmount)
  const orderDate = new Date(
    Date.UTC(
      2024 + (index % 3),
      rng.int(0, 11),
      rng.int(1, 28),
      rng.int(7, 20),
      rng.int(0, 59)
    )
  )
  const payment = choosePayment(rng)
  const orderStatus = chooseOrderStatus(rng, plan.blueprint)
  const serviceMode = chooseServiceMode(rng)
  const receiptType = chooseReceiptType(rng)
  const domain = `${slugify(sellerName)}.example`
  const record = {
    language: languageConfig.language,
    blueprint: plan.blueprint,
    currencyCode: currency.code,
    currencySymbol: currency.symbol,
    seller: {
      name: sellerName,
      telephone: generateTelephone(merchantAddress.addressCountry, rng),
      email: generateEmail(sellerName, domain, rng),
      taxId: generateTaxId(merchantAddress.addressCountry, rng),
      address: merchantAddress,
      website: `https://${domain}`,
    },
    customer: {
      type: rng.chance(0.8) ? 'Person' : 'Organization',
      name: customerName,
      email: generateEmail(customerName, 'customer.example', rng),
      address: customerAddress,
      identifier: `CUST-${rng.id(6)}`,
    },
    order: {
      number: generateOrderNumber(rng),
      dateIso: orderDate.toISOString().slice(0, 10),
      timeIso: orderDate.toISOString().slice(11, 16),
      paymentMethodDisplay: payment.display,
      paymentMethodUrl: payment.schemaUrl,
      paymentStatusDisplay: payment.paymentStatusDisplay,
      paymentStatusUrl: payment.paymentStatusUrl,
      orderStatusDisplay: orderStatus.display,
      orderStatusUrl: orderStatus.schemaUrl,
      receiptType,
      serviceMode,
      registerNumber: String(rng.int(1, 24)),
      cashierName: chooseBrokerName(languageConfig.values, rng),
      promoCode: discountAmount > 0 ? choosePromoCode(languageConfig.values, rng) : null,
    },
    delivery:
      plan.blueprint === 'shipping-notice' || rng.chance(0.35)
        ? {
            providerName: `${chooseMerchantName(rng)} Logistics`,
            trackingNumber: generateTrackingNumber(rng),
            address: customerAddress,
          }
        : null,
    items,
    amounts: {
      subtotal,
      discountAmount,
      taxAmount,
      total,
    },
    errors: {},
  }

  applyErrorProfile(record, rng, plan.errorProfile)
  return record
}

function applyErrorProfile(record, rng, errorProfile) {
  if (errorProfile === 'none') {
    return
  }

  const uncertainCandidates = [
    {
      path: 'seller.name',
      get: () => record.seller.name,
      set: (value) => {
        record.seller.name = value
      },
    },
    {
      path: 'seller.address.streetAddress',
      get: () => record.seller.address.streetAddress,
      set: (value) => {
        record.seller.address.streetAddress = value
        record.seller.address.fullText = rebuildAddress(record.seller.address)
      },
    },
    {
      path: 'customer.name',
      get: () => record.customer.name,
      set: (value) => {
        record.customer.name = value
      },
    },
    {
      path: 'items.0.name',
      get: () => record.items[0]?.name,
      set: (value) => {
        if (record.items[0]) {
          record.items[0].name = value
        }
      },
    },
  ]

  const fatalCandidates = [
    {
      path: 'order.number',
      get: () => record.order.number,
      set: (value) => {
        record.order.number = value
      },
    },
    {
      path: 'seller.telephone',
      get: () => record.seller.telephone,
      set: (value) => {
        record.seller.telephone = value
      },
    },
    {
      path: 'seller.email',
      get: () => record.seller.email,
      set: (value) => {
        record.seller.email = value
      },
    },
    {
      path: 'delivery.trackingNumber',
      get: () => record.delivery?.trackingNumber,
      set: (value) => {
        if (record.delivery) {
          record.delivery.trackingNumber = value
        }
      },
    },
  ].filter((candidate) => candidate.get())

  if (errorProfile === 'uncertain' || errorProfile === 'mixed') {
    const candidate = rng.pick(
      uncertainCandidates.filter((entry) => entry.get() && entry.get().length > 4)
    )
    const mutated = mutateText(candidate.get(), rng, 'uncertain')
    candidate.set(mutated)
    record.errors[candidate.path] = 'UNCERTAIN'
  }

  if (errorProfile === 'fatal' || errorProfile === 'mixed') {
    const candidate = rng.pick(
      fatalCandidates.filter((entry) => entry.get() && entry.get().length > 4)
    )
    const mutated = mutateText(candidate.get(), rng, 'fatal')
    candidate.set(mutated)
    record.errors[candidate.path] = 'FATAL'
  }
}

function mutateText(value, rng, mode) {
  const substitutions =
    mode === 'fatal'
      ? [
          ['0', 'O'],
          ['1', 'I'],
          ['5', 'S'],
          ['8', 'B'],
          ['@', 'a'],
          ['.', ','],
          ['-', ''],
        ]
      : [
          ['o', '0'],
          ['i', 'l'],
          ['e', 'c'],
          ['m', 'rn'],
          ['u', 'v'],
        ]

  let mutated = value
  const passes = mode === 'fatal' ? 3 : 1

  for (let index = 0; index < passes; index += 1) {
    const [source, target] = rng.pick(substitutions)
    if (mutated.includes(source)) {
      mutated = mutated.replace(source, target)
    } else {
      const position = rng.int(0, mutated.length - 1)
      mutated =
        mutated.slice(0, position) + target + mutated.slice(position + 1)
    }
  }

  return mutated
}

function rebuildAddress(address) {
  return [
    address.streetAddress,
    address.addressLocality,
    address.addressRegion,
    address.postalCode,
    address.addressCountry,
  ]
    .filter(Boolean)
    .join(', ')
}

function roundMoney(value) {
  return Number(value.toFixed(2))
}
