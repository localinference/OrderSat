import { applyNoise } from '../noise/applyNoise.js'
import { renderCsvExport } from './renderers/csvExport.js'
import { renderEmailSummary } from './renderers/emailSummary.js'
import { renderHtmlOrder } from './renderers/htmlOrder.js'
import { renderJsonDump } from './renderers/jsonDump.js'
import { renderPlainReceipt } from './renderers/plainReceipt.js'
import { renderXmlSummary } from './renderers/xmlSummary.js'

const rendererMap = {
  'plain-receipt': renderPlainReceipt,
  'email-summary': renderEmailSummary,
  'html-order': renderHtmlOrder,
  'json-dump': renderJsonDump,
  'xml-summary': renderXmlSummary,
  'csv-export': renderCsvExport,
}

export function renderInput({ record, labels, rng, plan }) {
  const renderer = rendererMap[plan.renderer]
  if (!renderer) {
    throw new Error(`Unsupported renderer "${plan.renderer}".`)
  }

  const rendered = renderer(record, labels, rng)
  return applyNoise(rendered, rng, plan.noise)
}
