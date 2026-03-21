import { expandTemplatedValue, splitAddress } from './placeholders.js'

const merchantPrefixes = [
  'Blue',
  'Prime',
  'Urban',
  'Golden',
  'North',
  'Green',
  'Silver',
  'Modern',
  'Bright',
  'Fresh',
  'Summit',
  'Harbor',
  'Oak',
  'Maple',
  'True',
]

const merchantRoots = [
  'Market',
  'Supply',
  'Depot',
  'Kitchen',
  'Studio',
  'Works',
  'Collective',
  'Center',
  'Hub',
  'Services',
  'Store',
  'Outfitters',
  'Corner',
  'House',
  'Goods',
]

const merchantSuffixes = ['Ltd', 'Co', 'Group', 'Partners', 'Shop', 'Labs']

const paymentMethods = [
  { display: 'Cash', schemaUrl: 'https://schema.org/Cash' },
  { display: 'Visa', schemaUrl: 'https://schema.org/CreditCard' },
  { display: 'Mastercard', schemaUrl: 'https://schema.org/CreditCard' },
  {
    display: 'Bank Transfer',
    schemaUrl: 'https://schema.org/ByBankTransferInAdvance',
  },
]

const orderStatuses = [
  { display: 'Processing', schemaUrl: 'https://schema.org/OrderProcessing' },
  { display: 'In Transit', schemaUrl: 'https://schema.org/OrderInTransit' },
  { display: 'Delivered', schemaUrl: 'https://schema.org/OrderDelivered' },
  {
    display: 'Pickup Available',
    schemaUrl: 'https://schema.org/OrderPickupAvailable',
  },
]

const paymentStatuses = [
  {
    display: 'Paid',
    schemaUrl: 'https://schema.org/PaymentComplete',
  },
  {
    display: 'Due',
    schemaUrl: 'https://schema.org/PaymentDue',
  },
]

const serviceModes = ['Dine In', 'Takeaway', 'Delivery', 'Pickup', 'Online']
const receiptTypes = [
  'Sales Receipt',
  'Order Confirmation',
  'Checkout Summary',
  'Shipment Notice',
  'Service Confirmation',
]

export function chooseCustomerName(values, rng) {
  return rng.pick(values.customer)
}

export function chooseBrokerName(values, rng) {
  return rng.pick(values.broker)
}

export function chooseMerchantName(rng) {
  return `${rng.pick(merchantPrefixes)} ${rng.pick(merchantRoots)} ${rng.pick(merchantSuffixes)}`
}

export function chooseAddress(values, rng) {
  const fullText = expandTemplatedValue(rng.pick(values.billingAddress), rng)
  return splitAddress(fullText)
}

export function chooseItems(values, rng, blueprint) {
  const itemCount =
    blueprint === 'service-confirmation' ? rng.int(1, 3) : rng.int(1, 5)

  const names = rng.shuffle(values.acceptedOffer).slice(0, itemCount)
  return names.map((name) => {
    const type = inferOfferType(name, blueprint)
    const quantity = type === 'Service' ? 1 : rng.int(1, 4)
    const unitPrice = randomMoney(
      rng,
      type === 'Service' ? 18 : 4,
      type === 'Service' ? 180 : 120
    )
    const linePrice = quantity * unitPrice

    return {
      type,
      name,
      sku: `${rng.id(3)}-${rng.id(4)}`,
      quantity,
      unitText: 'item',
      unitPrice,
      linePrice,
    }
  })
}

export function chooseCurrency(country, rng) {
  if (country === 'United Kingdom') {
    return { code: 'GBP', symbol: '£' }
  }
  if (country === 'Ireland' || country === 'France' || country === 'Germany') {
    return { code: 'EUR', symbol: '€' }
  }
  if (country === 'Canada') {
    return { code: 'CAD', symbol: 'CA$' }
  }
  if (country === 'Australia') {
    return { code: 'AUD', symbol: 'A$' }
  }
  return rng.pick([
    { code: 'USD', symbol: '$' },
    { code: 'EUR', symbol: '€' },
    { code: 'GBP', symbol: '£' },
  ])
}

export function choosePayment(rng) {
  const method = rng.pick(paymentMethods)
  const status = rng.weightedPick([
    { value: paymentStatuses[0], weight: 4 },
    { value: paymentStatuses[1], weight: 1 },
  ])

  return {
    ...method,
    paymentStatusDisplay: status.display,
    paymentStatusUrl: status.schemaUrl,
  }
}

export function chooseOrderStatus(rng, blueprint) {
  if (blueprint === 'shipping-notice') {
    return rng.weightedPick([
      { value: orderStatuses[1], weight: 2 },
      { value: orderStatuses[2], weight: 1 },
    ])
  }

  if (blueprint === 'service-confirmation') {
    return orderStatuses[0]
  }

  return rng.weightedPick([
    { value: orderStatuses[0], weight: 3 },
    { value: orderStatuses[3], weight: 1 },
  ])
}

export function generateOrderNumber(rng) {
  return `${rng.id(2)}-${rng.id(4)}-${rng.id(4)}`
}

export function generateTrackingNumber(rng) {
  return `${rng.id(4)}${rng.id(4)}${rng.id(4)}${rng.id(4)}`
}

export function generateTelephone(country, rng) {
  if (country === 'United Kingdom') {
    return `+44 20 ${rng.int(1000, 9999)} ${rng.int(1000, 9999)}`
  }
  if (country === 'Ireland') {
    return `+353 1 ${rng.int(100, 999)} ${rng.int(1000, 9999)}`
  }
  return `+1 ${rng.int(200, 999)}-${rng.int(200, 999)}-${rng.int(1000, 9999)}`
}

export function generateEmail(name, domain, rng) {
  const slug = slugify(name).replaceAll('-', '.')
  return `${slug}${rng.int(1, 99)}@${domain}`
}

export function generateTaxId(country, rng) {
  if (country === 'United Kingdom') {
    return `GB${rng.int(100000000, 999999999)}`
  }
  if (country === 'Ireland') {
    return `${rng.id(7)}${rng.pick(['W', 'X'])}`
  }
  if (country === 'France') {
    return `FR${rng.int(10000000000, 99999999999)}`
  }
  return `${rng.id(2)}-${rng.id(7)}`
}

export function choosePromoCode(values, rng) {
  return rng.pick(values.discountCode)
}

export function chooseServiceMode(rng) {
  return rng.pick(serviceModes)
}

export function chooseReceiptType(rng) {
  return rng.pick(receiptTypes)
}

export function randomMoney(rng, min, max) {
  return Number((rng.int(min * 100, max * 100) / 100).toFixed(2))
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

function inferOfferType(name, blueprint) {
  if (blueprint === 'service-confirmation') {
    return 'Service'
  }

  return /service|session|subscription|ticket|pass|booking|consultation|lesson|plan|access/i.test(
    name
  )
    ? 'Service'
    : 'Product'
}
