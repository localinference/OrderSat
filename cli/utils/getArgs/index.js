import { parseArgs } from 'util'
import { arrayFromCommas } from '../arrayFromCommas/index.js'

/**
 * @returns {{languages: Array<string>}}
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
    },
  })
  return {
    languages: arrayFromCommas(values.languages),
  }
}
