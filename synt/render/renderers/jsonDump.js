import { formatAddress, formatCurrency, formatDateTime } from '../helpers.js'

export function renderJsonDump(record, _labels, rng) {
  const view = {
    seller_name: record.seller.name,
    seller_contact: {
      phone: record.seller.telephone,
      email: record.seller.email,
      address: formatAddress(record.seller.address),
    },
    order: {
      id: record.order.number,
      placed_at: formatDateTime(record, rng),
      customer: record.customer.name,
      status: record.order.orderStatusDisplay,
      payment: record.order.paymentMethodDisplay,
      currency: record.currencyCode,
    },
    lines: record.items.map((item) => ({
      name: item.name,
      qty: item.quantity,
      unit_price: formatCurrency(item.unitPrice, record, rng),
      line_total: formatCurrency(item.linePrice, record, rng),
    })),
    totals: {
      subtotal: formatCurrency(record.amounts.subtotal, record, rng),
      tax: formatCurrency(record.amounts.taxAmount, record, rng),
      total: formatCurrency(record.amounts.total, record, rng),
    },
  }

  if (record.delivery) {
    view.delivery = {
      provider: record.delivery.providerName,
      shipped_date: record.delivery.shippedDateIso,
      address: formatAddress(record.delivery.address),
      tracking: record.delivery.trackingNumber,
    }
  }

  return JSON.stringify(view, null, 2)
}
