import { mkdir } from 'fs/promises'

export async function ensurePath(path) {
  await mkdir(path, { recursive: true })
  return path
}
