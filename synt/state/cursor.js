import fs from 'fs/promises'
import { dirname } from 'path'

const cursorFilePath = './tmp/synt-state/cursors.json'

export async function readCursor({ language, seed }) {
  const state = await readState()
  return state[getCursorKey(language, seed)] ?? 0
}

export async function writeCursor({ language, seed, nextIndex }) {
  const state = await readState()
  const key = getCursorKey(language, seed)
  const previous = state[key] ?? 0
  state[key] = Math.max(previous, nextIndex)

  await fs.mkdir(dirname(cursorFilePath), { recursive: true })
  await fs.writeFile(
    cursorFilePath,
    `${JSON.stringify(state, null, 2)}\n`,
    'utf-8'
  )
}

function getCursorKey(language, seed) {
  return `${language}:${seed}`
}

async function readState() {
  try {
    const raw = await fs.readFile(cursorFilePath, 'utf-8')
    const parsed = JSON.parse(raw)
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new Error('Invalid cursor state shape.')
    }
    return parsed
  } catch (error) {
    if (error?.code === 'ENOENT') {
      return {}
    }

    throw error
  }
}
