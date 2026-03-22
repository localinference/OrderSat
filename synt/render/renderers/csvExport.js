import { formatAddress, formatCurrency, formatDateTime } from '../helpers.js'

export function renderCsvExport(record, _labels, rng) {
  const rows = [
    ['section', 'field', 'value'],
    ['seller', 'name', record.seller.name],
    ['seller', 'address', formatAddress(record.seller.address)],
    ['seller', 'telephone', record.seller.telephone],
    ['seller', 'email', record.seller.email],
    ['order', 'order_number', record.order.number],
    ['order', 'order_datetime', formatDateTime(record, rng)],
    ['order', 'customer', record.customer.name],
    ['order', 'payment_method', record.order.paymentMethodDisplay],
    ['order', 'status', record.order.orderStatusDisplay],
  ]

  for (const [index, item] of record.items.entries()) {
    rows.push(
      ['line', `item_${index + 1}_name`, item.name],
      ['line', `item_${index + 1}_qty`, String(item.quantity)],
      [
        'line',
        `item_${index + 1}_unit_price`,
        formatCurrency(item.unitPrice, record, rng),
      ],
      [
        'line',
        `item_${index + 1}_total`,
        formatCurrency(item.linePrice, record, rng),
      ]
    )
  }

  rows.push(
    ['totals', 'subtotal', formatCurrency(record.amounts.subtotal, record, rng)],
    ['totals', 'tax', formatCurrency(record.amounts.taxAmount, record, rng)],
    ['totals', 'total', formatCurrency(record.amounts.total, record, rng)]
  )

  if (record.delivery) {
    rows.push(
      ['delivery', 'provider', record.delivery.providerName],
      ['delivery', 'shipped_date', record.delivery.shippedDateIso],
      ['delivery', 'address', formatAddress(record.delivery.address)],
      ['delivery', 'tracking', record.delivery.trackingNumber]
    )
  }

  return rows.map((row) => row.map(escapeCsv).join(',')).join('\n')
}

function escapeCsv(value) {
  const text = String(value)
  if (/[",\n]/.test(text)) {
    return `"${text.replaceAll('"', '""')}"`
  }
  return text
}
