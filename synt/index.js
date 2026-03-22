#!/usr/bin/env node

import { parseCliArgs } from './args/parse.js'
import { createWorkerPool } from './runWithWorker/index.js'
import { pairHash } from './hash/pairHash.js'
import { writeSamplePair } from './io/writeSamplePair.js'
import { readCursor, writeCursor } from './state/cursor.js'
import { validateStructure } from '../cli/utils/validateStructure/index.js'

const inputRoot = './src/02_training_samples/inputs'
const outputRoot = './src/02_training_samples/outputs'

try {
  const options = parseCliArgs()
  const workerPool = createWorkerPool(options.concurrency)

  try {
    for (const language of options.languages) {
      await generateLanguageSamples({
        language,
        options,
        workerPool,
      })
    }
  } finally {
    await workerPool.close()
  }
} catch (error) {
  console.error(error?.stack ?? String(error))
  process.exitCode = 1
}

async function generateLanguageSamples({ language, options, workerPool }) {
  const startedAt = performance.now()
  const validatedCoverageKeys = new Set()
  const cursorStart = await readCursor({
    language,
    seed: options.seed,
  })
  const maxAttempts = Math.max(
    options.count,
    options.count * options.maxAttemptsFactor
  )
  const maxIndexExclusive = cursorStart + maxAttempts

  let nextIndex = cursorStart
  let attempted = 0
  let written = 0
  let skipped = 0
  let validated = 0
  let maxAttemptedIndexExclusive = cursorStart
  let writeLock = Promise.resolve()

  console.log(
    `[synt] Starting synthetic generation for "${language}". Target new samples: ${options.count}. Concurrency: ${options.concurrency}. Batch size: ${options.batchSize}. Seed: ${options.seed}. Cursor: ${cursorStart}.\n`
  )

  await Promise.all(
    Array.from({ length: options.concurrency }, async () => {
      for (;;) {
        if (written >= options.count) {
          return
        }

        const startIndex = nextIndex
        if (startIndex >= maxIndexExclusive) {
          return
        }

        nextIndex += options.batchSize

        const requestCount = Math.min(
          options.batchSize,
          maxIndexExclusive - startIndex
        )

        const { samples } = await workerPool.run('generateBatch', {
          language,
          startIndex,
          count: requestCount,
          baseSeed: options.seed,
        })

        for (const sample of samples) {
          attempted += 1
          maxAttemptedIndexExclusive = Math.max(
            maxAttemptedIndexExclusive,
            sample.absoluteIndex + 1
          )

          if (written >= options.count) {
            return
          }

          if (
            shouldValidateSample(
              sample.coverageKey,
              options.validateMode,
              validatedCoverageKeys
            )
          ) {
            const validation = await validateStructure(sample.outputText)
            if (validation !== true) {
              throw new Error(
                `Synthetic sample validation failed for language "${language}" at coverage "${sample.coverageKey}": ${JSON.stringify(validation, null, 2)}`
              )
            }
            validatedCoverageKeys.add(sample.coverageKey)
            validated += 1
          }

          const fileStem = pairHash(sample.inputText, sample.outputStableText)
          const result = await withWriteLock(async () => {
            if (written >= options.count) {
              return { written: false, skipped: false, saturated: true }
            }

            const writeResult = await writeSamplePair({
              fileStem,
              inputRoot,
              outputRoot,
              language,
              inputText: sample.inputText,
              outputText: sample.outputText,
            })

            if (writeResult.written) {
              written += 1
              return { written: true, skipped: false, saturated: false }
            }

            skipped += 1
            return { written: false, skipped: true, saturated: false }
          })

          if (result.saturated) {
            return
          }
        }
      }
    })
  )

  if (written < options.count) {
    throw new Error(
      `Unable to write the requested ${options.count} new "${language}" synthetic samples within ${maxAttempts} attempts starting from cursor ${cursorStart}. Wrote ${written}, skipped ${skipped}.`
    )
  }

  await writeCursor({
    language,
    seed: options.seed,
    nextIndex: maxAttemptedIndexExclusive,
  })

  console.log(
    `[synt] Finished "${language}". Wrote ${written} new pairs, skipped ${skipped} existing pairs, attempted ${attempted} candidates, validated ${validated} structural sample(s), advanced cursor to ${maxAttemptedIndexExclusive} in ${Math.round(performance.now() - startedAt)} ms.\n`
  )

  function withWriteLock(task) {
    const run = writeLock.then(task)
    writeLock = run.catch(() => {})
    return run
  }
}

function shouldValidateSample(coverageKey, validateMode, validatedCoverageKeys) {
  if (validateMode === 'none') {
    return false
  }

  if (validateMode === 'all') {
    return true
  }

  return !validatedCoverageKeys.has(coverageKey)
}
