import {
  formatAddress,
  formatCurrency,
  formatDateTime,
  formatField,
  pickLabel,
} from '../helpers.js'

export function renderPlainReceipt(record, labels, rng) {
  const lines = [
    record.seller.name,
    formatAddress(record.seller.address),
    formatField(
      pickLabel(labels, 'telephone', rng),
      record.seller.telephone,
      rng
    ),
    formatField(pickLabel(labels, 'email', rng), record.seller.email, rng),
    '',
    formatField(pickLabel(labels, 'orderNumber', rng), record.order.number, rng),
    formatField(
      pickLabel(labels, 'orderDate', rng),
      formatDateTime(record, rng),
      rng
    ),
    formatField(pickLabel(labels, 'customer', rng), record.customer.name, rng),
    formatField(
      pickLabel(labels, 'paymentMethod', rng),
      record.order.paymentMethodDisplay,
      rng
    ),
    '',
    rng.pick(labels.acceptedOffer),
    '----------------------------------------',
  ]

  for (const item of record.items) {
    lines.push(
      `${item.name}`,
      `  ${item.quantity} x ${formatCurrency(item.unitPrice, record, rng)} = ${formatCurrency(item.linePrice, record, rng)}`
    )
  }

  lines.push(
    '----------------------------------------',
    formatField(
      pickLabel(labels, 'subtotal', rng),
      formatCurrency(record.amounts.subtotal, record, rng),
      rng
    ),
    formatField(
      pickLabel(labels, 'taxAmount', rng),
      formatCurrency(record.amounts.taxAmount, record, rng),
      rng
    )
  )

  if (record.amounts.discountAmount > 0 && record.order.promoCode) {
    lines.push(
      formatField(
        pickLabel(labels, 'discountCode', rng),
        record.order.promoCode,
        rng
      ),
      formatField(
        pickLabel(labels, 'discountAmount', rng),
        formatCurrency(record.amounts.discountAmount, record, rng),
        rng
      )
    )
  }

  lines.push(
    formatField(
      pickLabel(labels, 'totalPaymentDue', rng),
      formatCurrency(record.amounts.total, record, rng),
      rng
    ),
    formatField(
      pickLabel(labels, 'cashierName', rng),
      record.order.cashierName,
      rng
    ),
    formatField(
      pickLabel(labels, 'registerNumber', rng),
      record.order.registerNumber,
      rng
    ),
    formatField(
      pickLabel(labels, 'serviceMode', rng),
      record.order.serviceMode,
      rng
    ),
    formatField(
      pickLabel(labels, 'receiptType', rng),
      record.order.receiptType,
      rng
    )
  )

  if (record.delivery) {
    lines.push(
      '',
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

  return lines.join('\n')
}
