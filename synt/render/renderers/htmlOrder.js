import {
  escapeHtml,
  formatAddress,
  formatCurrency,
  formatDateTime,
  pickLabel,
} from '../helpers.js'

export function renderHtmlOrder(record, labels, rng) {
  const itemRows = record.items
    .map(
      (item) =>
        `<tr><td>${escapeHtml(item.name)}</td><td>${item.quantity}</td><td>${escapeHtml(
          formatCurrency(item.unitPrice, record, rng)
        )}</td><td>${escapeHtml(formatCurrency(item.linePrice, record, rng))}</td></tr>`
    )
    .join('')

  return [
    '<section class="order">',
    `<h1>${escapeHtml(record.seller.name)}</h1>`,
    `<p>${escapeHtml(formatAddress(record.seller.address))}</p>`,
    `<p>${escapeHtml(pickLabel(labels, 'orderNumber', rng))}: ${escapeHtml(
      record.order.number
    )}</p>`,
    `<p>${escapeHtml(pickLabel(labels, 'orderDate', rng))}: ${escapeHtml(
      formatDateTime(record, rng)
    )}</p>`,
    `<p>${escapeHtml(pickLabel(labels, 'customer', rng))}: ${escapeHtml(
      record.customer.name
    )}</p>`,
    `<table><thead><tr><th>Item</th><th>Qty</th><th>Unit</th><th>Total</th></tr></thead><tbody>${itemRows}</tbody></table>`,
    `<p>${escapeHtml(pickLabel(labels, 'totalPaymentDue', rng))}: ${escapeHtml(
      formatCurrency(record.amounts.total, record, rng)
    )}</p>`,
    `<p>${escapeHtml(pickLabel(labels, 'paymentMethod', rng))}: ${escapeHtml(
      record.order.paymentMethodDisplay
    )}</p>`,
    '</section>',
  ].join('\n')
}
