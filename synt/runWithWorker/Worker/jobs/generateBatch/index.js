import { loadLanguageConfig } from '../../../../config/load/index.js'
import {
  stablePrettyStringify,
  stableStringify,
} from '../../../../json/stable.js'
import { createRng } from '../../../../random/createRng.js'
import { renderInput } from '../../../../render/index.js'
import {
  buildOrderRecord,
  getCoveragePlan,
} from '../../../../semantic/buildOrderRecord.js'
import { buildJsonLd } from '../../../../semantic/buildJsonLd.js'

export async function generateBatch({ language, startIndex, count, baseSeed }) {
  const languageConfig = await loadLanguageConfig(language)
  const samples = []

  for (let offset = 0; offset < count; offset += 1) {
    const absoluteIndex = startIndex + offset
    const plan = getCoveragePlan(absoluteIndex)
    const rng = createRng(
      `${language}:${baseSeed}:${absoluteIndex}:${plan.coverageKey}`
    )
    const record = buildOrderRecord({
      languageConfig,
      rng,
      plan,
      index: absoluteIndex,
    })
    const output = buildJsonLd(record)

    samples.push({
      absoluteIndex,
      coverageKey: plan.coverageKey,
      inputText: renderInput({
        record,
        labels: languageConfig.labels,
        rng,
        plan,
      }),
      outputStableText: stableStringify(output),
      outputText: stablePrettyStringify(output),
    })
  }

  return { samples }
}
