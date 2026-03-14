import { parentPort, workerData } from 'worker_threads'
import { unpack } from '@nodearchive/nodearchive'

try {
  const result = await unpack({
    literalPath: workerData.route,
    destinationPath: workerData.destinationPath,
    force: true,
    passThru: true,
  })

  parentPort.postMessage(result)
} catch (error) {
  throw error
}
