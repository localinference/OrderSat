import { unpack } from '@nodearchive/nodearchive'
export async function unpackArchive({ path, dest }) {
  try {
    console.log(`[Archive] Unpacking "${path}" into "${dest}".\n`)
    const result = await unpack({
      literalPath: path,
      destinationPath: dest,
      force: true,
      passThru: true,
    })
    console.log(`[Archive] Finished unpacking "${path}".\n`)
    return result
  } catch (err) {
    throw err
  }
}
