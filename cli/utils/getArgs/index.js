import { parseArgs } from 'util'
import { arrayFromCommas } from '../arrayFromCommas/index.js'

/**
 * @returns {{language: Array<string>, destinationPath: import('fs').PathLike}}
 */
export function getArgs() {
  const { values } = parseArgs({
    options: {
      languages: {
        default: 'eng',
        short: 'L',
        type: 'string',
        multiple: false,
      },
      destinationPath: {
        type: 'string',
        short: 'D',
        multiple: false,
        default: './models/training_samples/inputs',
      },
    },
  })
  return {
    languages: arrayFromCommas(values.languages),
    destinationPath: values.destinationPath,
  }
}
