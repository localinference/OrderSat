import os from 'os'
import { parseArgs } from 'util'
import { arrayFromCommas } from '../../cli/utils/arrayFromCommas/index.js'

export function parseCliArgs() {
  const { values } = parseArgs({
    options: {
      languages: {
        short: 'L',
        type: 'string',
        default: 'eng',
      },
      count: {
        short: 'C',
        type: 'string',
        default: '1000',
      },
      batchSize: {
        type: 'string',
        default: '64',
      },
      concurrency: {
        type: 'string',
        default: String(Math.max(1, Math.min(8, os.availableParallelism()))),
      },
      seed: {
        type: 'string',
        default: '1',
      },
      validateMode: {
        type: 'string',
        default: 'sample',
      },
      maxAttemptsFactor: {
        type: 'string',
        default: '4',
      },
    },
  })

  return {
    languages: arrayFromCommas(values.languages),
    count: parsePositiveInteger(values.count, 'count'),
    batchSize: parsePositiveInteger(values.batchSize, 'batchSize'),
    concurrency: parsePositiveInteger(values.concurrency, 'concurrency'),
    seed: values.seed,
    validateMode: parseValidateMode(values.validateMode),
    maxAttemptsFactor: parsePositiveInteger(
      values.maxAttemptsFactor,
      'maxAttemptsFactor'
    ),
  }
}

function parsePositiveInteger(value, name) {
  const parsed = Number.parseInt(value, 10)
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`Invalid --${name} value "${value}". Expected a positive integer.`)
  }
  return parsed
}

function parseValidateMode(value) {
  if (value === 'all' || value === 'sample' || value === 'none') {
    return value
  }

  throw new Error(
    `Invalid --validateMode value "${value}". Expected one of: all, sample, none.`
  )
}
