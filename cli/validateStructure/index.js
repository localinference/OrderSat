import { validateStructure } from '../utils/validateStructure/index.js'
import { parseArgs } from 'util'
import fs from 'fs/promises'

const { positionals } = parseArgs({
  allowPositionals: true,
})

const content = await fs.readFile(positionals[0], { encoding: 'utf-8' })
const valid = await validateStructure(content)
console.log(valid)
