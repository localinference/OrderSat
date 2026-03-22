import {
  escapeXml,
  formatAddress,
  formatCurrency,
  formatDateTime,
} from '../helpers.js'

export function renderXmlSummary(record, _labels, rng) {
  const items = record.items
    .map(
      (item) =>
        `<item><name>${escapeXml(item.name)}</name><quantity>${item.quantity}</quantity><unitPrice>${escapeXml(
          formatCurrency(item.unitPrice, record, rng)
        )}</unitPrice><linePrice>${escapeXml(
          formatCurrency(item.linePrice, record, rng)
        )}</linePrice></item>`
    )
    .join('')

  return [
    '<orderDocument>',
    `<seller name="${escapeXml(record.seller.name)}" phone="${escapeXml(
      record.seller.telephone
    )}" email="${escapeXml(record.seller.email)}">`,
    `<address>${escapeXml(formatAddress(record.seller.address))}</address>`,
    '</seller>',
    `<order id="${escapeXml(record.order.number)}" placedAt="${escapeXml(
      formatDateTime(record, rng)
    )}" payment="${escapeXml(record.order.paymentMethodDisplay)}" status="${escapeXml(
      record.order.orderStatusDisplay
    )}">`,
    `<customer>${escapeXml(record.customer.name)}</customer>`,
    `<items>${items}</items>`,
    `<totals subtotal="${escapeXml(
      formatCurrency(record.amounts.subtotal, record, rng)
    )}" tax="${escapeXml(
      formatCurrency(record.amounts.taxAmount, record, rng)
    )}" total="${escapeXml(formatCurrency(record.amounts.total, record, rng))}" />`,
    ...(record.delivery
      ? [
          `<delivery provider="${escapeXml(
            record.delivery.providerName
          )}" tracking="${escapeXml(
            record.delivery.trackingNumber
          )}" shippedDate="${escapeXml(record.delivery.shippedDateIso)}">`,
          `<address>${escapeXml(formatAddress(record.delivery.address))}</address>`,
          '</delivery>',
        ]
      : []),
    '</order>',
    '</orderDocument>',
  ].join('\n')
}
