import {
  formatAddress,
  formatCurrency,
  formatDate,
  formatField,
  pickLabel,
} from '../helpers.js'

export function renderEmailSummary(record, labels, rng) {
  const lines = [
    `Subject: ${record.order.receiptType} ${record.order.number}`,
    '',
    `Hello ${record.customer.name},`,
    '',
    `Your order with ${record.seller.name} is ${record.order.orderStatusDisplay.toLowerCase()}.`,
    formatField(pickLabel(labels, 'orderDate', rng), formatDate(record, rng), rng),
    formatField(
      pickLabel(labels, 'paymentMethod', rng),
      record.order.paymentMethodDisplay,
      rng
    ),
    '',
    'Items:',
    ...record.items.map(
      (item) =>
        `- ${item.name} (${item.quantity} x ${formatCurrency(item.unitPrice, record, rng)}) => ${formatCurrency(item.linePrice, record, rng)}`
    ),
    '',
    formatField(
      pickLabel(labels, 'totalPaymentDue', rng),
      formatCurrency(record.amounts.total, record, rng),
      rng
    ),
  ]

  if (record.delivery) {
    lines.push(
      formatField(
        pickLabel(labels, 'shippingAddress', rng),
        formatAddress(record.delivery.address),
        rng
      ),
      formatField(
        pickLabel(labels, 'trackingNumber', rng),
        record.delivery.trackingNumber,
        rng
      )
    )
  }

  lines.push(
    '',
    `Seller contact: ${record.seller.telephone} / ${record.seller.email}`,
    `Customer ID: ${record.customer.identifier}`
  )

  return lines.join('\n')
}
