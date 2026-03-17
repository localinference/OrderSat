import FastGlob from 'fast-glob'

export async function getGlobLength(glob) {
  const files = await FastGlob.async(glob, {
    dot: true,
    onlyFiles: true,
  })

  return files.length
}
