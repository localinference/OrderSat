import { toMoneyString } from '../values/factories.js'

export function pickLabel(labels, key, rng) {
  const candidates = labels[key]
  if (Array.isArray(candidates) && candidates.length > 0) {
    return rng.pick(candidates)
  }

  return humanizeKey(key)
}

export function formatField(label, value, rng) {
  const separator = rng.pick([': ', ' - ', ' = '])
  return `${label}${separator}${value}`
}

export function formatCurrency(value, record, rng) {
  const number = toMoneyString(value)
  return rng.pick([
    `${record.currencySymbol}${number}`,
    `${record.currencyCode} ${number}`,
    `${number} ${record.currencyCode}`,
  ])
}

export function formatDate(record, rng) {
  const [year, month, day] = record.order.dateIso.split('-')
  return rng.pick([
    `${year}-${month}-${day}`,
    `${day}/${month}/${year}`,
    `${month}/${day}/${year}`,
    `${day}.${month}.${year}`,
  ])
}

export function formatDateTime(record, rng) {
  const date = formatDate(record, rng)
  return rng.pick([
    `${date} ${record.order.timeIso}`,
    `${date} ${record.order.timeIso}:00`,
    `${date}T${record.order.timeIso}`,
  ])
}

export function formatAddress(address) {
  return address.fullText
}

export function escapeHtml(value) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
}

export function escapeXml(value) {
  return escapeHtml(value).replaceAll('"', '&quot;')
}

function humanizeKey(value) {
  return value
    .replaceAll(/([a-z])([A-Z])/g, '$1 $2')
    .replaceAll(/[_-]+/g, ' ')
    .toLowerCase()
}
