import { unpack } from '@nodearchive/nodearchive'
export async function unpackArchive({ path, dest }) {
  try {
    return await unpack({
      literalPath: path,
      destinationPath: dest,
      force: true,
      passThru: true,
    })
  } catch (err) {
    throw err
  }
}
