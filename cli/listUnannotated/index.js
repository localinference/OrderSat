import { copyFile, mkdir, readdir, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { basename, extname, join, resolve } from 'node:path'

const cwd = process.cwd()

async function main() {
  const { language, previewCount } = parseArgs(process.argv.slice(2))
  const inputsDir = resolve(cwd, 'src/02_training_samples/inputs', language)
  const outputsDir = resolve(cwd, 'src/02_training_samples/outputs', language)
  const stagingDir = resolve(cwd, 'temp-unannotated', language)
  const manifestPath = join(stagingDir, 'manifest.json')

  await mkdir(stagingDir, { recursive: true })

  const inputFiles = await listFiles(inputsDir, '.txt')
  const outputFiles = await listFiles(outputsDir, '.jsonld')
  const inputBases = new Map(inputFiles.map((file) => [stripExt(file), file]))
  const outputBases = new Set(outputFiles.map(stripExt))
  const unannotatedBases = [...inputBases.keys()]
    .filter((base) => !outputBases.has(base))
    .sort()

  await pruneStaging(stagingDir, new Set(unannotatedBases))
  await syncStaging(inputsDir, stagingDir, inputBases, unannotatedBases)
  await writeManifest(manifestPath, {
    language,
    inputCount: inputFiles.length,
    outputCount: outputFiles.length,
    unannotatedCount: unannotatedBases.length,
    generatedAt: new Date().toISOString(),
    files: unannotatedBases,
  })

  printSummary({
    language,
    inputsDir,
    outputsDir,
    stagingDir,
    inputCount: inputFiles.length,
    outputCount: outputFiles.length,
    unannotatedBases,
    previewCount,
  })
}

function parseArgs(argv) {
  let language = 'eng'
  let previewCount = 20

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index]
    if (token === '-L' || token === '--language') {
      language = argv[index + 1] ?? language
      index += 1
      continue
    }
    if (token === '-n' || token === '--preview-count') {
      previewCount = Number(argv[index + 1] ?? previewCount)
      index += 1
      continue
    }
    if (token === '-h' || token === '--help') {
      printHelp()
      process.exit(0)
    }
    throw new Error(`Unknown argument: ${token}`)
  }

  if (!language.trim()) {
    throw new Error('Language cannot be empty.')
  }

  if (!Number.isInteger(previewCount) || previewCount < 0) {
    throw new Error('Preview count must be a non-negative integer.')
  }

  return {
    language,
    previewCount,
  }
}

async function listFiles(dirPath, expectedExtension) {
  const entries = await readdir(dirPath, { withFileTypes: true })
  return entries
    .filter(
      (entry) => entry.isFile() && extname(entry.name).toLowerCase() === expectedExtension
    )
    .map((entry) => entry.name)
    .sort()
}

async function pruneStaging(stagingDir, activeBases) {
  const entries = await readdir(stagingDir, { withFileTypes: true })

  await Promise.all(
    entries.map(async (entry) => {
      if (entry.name === 'manifest.json') {
        return
      }
      if (!entry.isFile() || extname(entry.name).toLowerCase() !== '.txt') {
        await rm(join(stagingDir, entry.name), { recursive: true, force: true })
        return
      }

      if (!activeBases.has(stripExt(entry.name))) {
        await rm(join(stagingDir, entry.name), { force: true })
      }
    })
  )
}

async function syncStaging(inputsDir, stagingDir, inputBases, unannotatedBases) {
  for (const base of unannotatedBases) {
    const fileName = inputBases.get(base)
    const sourcePath = join(inputsDir, fileName)
    const targetPath = join(stagingDir, fileName)
    const shouldCopy = await needsCopy(sourcePath, targetPath)

    if (shouldCopy) {
      await copyFileWithRetry(sourcePath, targetPath)
    }
  }
}

async function copyFileWithRetry(sourcePath, targetPath, attempts = 5) {
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      await copyFile(sourcePath, targetPath)
      return
    } catch (error) {
      if (!isTransientCopyError(error) || attempt === attempts) {
        throw error
      }
      await sleep(100 * attempt)
    }
  }
}

async function needsCopy(sourcePath, targetPath) {
  try {
    const [sourceStats, targetStats] = await Promise.all([
      stat(sourcePath),
      stat(targetPath),
    ])

    if (sourceStats.size !== targetStats.size) {
      return true
    }

    if (sourceStats.mtimeMs !== targetStats.mtimeMs) {
      const [sourceContent, targetContent] = await Promise.all([
        readFile(sourcePath, 'utf8'),
        readFile(targetPath, 'utf8'),
      ])
      return sourceContent !== targetContent
    }

    return false
  } catch {
    return true
  }
}

async function writeManifest(manifestPath, manifest) {
  await writeFile(`${manifestPath}`, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8')
}

function stripExt(fileName) {
  return basename(fileName, extname(fileName))
}

function printSummary({
  language,
  inputsDir,
  outputsDir,
  stagingDir,
  inputCount,
  outputCount,
  unannotatedBases,
  previewCount,
}) {
  console.log('[listUnannotated] Sync complete.')
  console.log(`Language: ${language}`)
  console.log(`Inputs: ${inputsDir}`)
  console.log(`Outputs: ${outputsDir}`)
  console.log(`Temp: ${stagingDir}`)
  console.log(`Input count: ${inputCount}`)
  console.log(`Output count: ${outputCount}`)
  console.log(`Unannotated count: ${unannotatedBases.length}`)

  if (unannotatedBases.length === 0) {
    console.log('No unannotated samples remain.')
    return
  }

  console.log('')
  console.log(`Next ${Math.min(previewCount, unannotatedBases.length)} unannotated sample(s):`)
  for (const base of unannotatedBases.slice(0, previewCount)) {
    console.log(`- ${base}.txt`)
  }
}

function printHelp() {
  console.log('Usage: node cli/listUnannotated [--language eng] [--preview-count 20]')
  console.log('')
  console.log('Mirrors src/02_training_samples input-minus-output diff into temp-unannotated/<language>.')
}

function isTransientCopyError(error) {
  return error?.code === 'EBUSY' || error?.code === 'EPERM'
}

function sleep(ms) {
  return new Promise((resolvePromise) => {
    setTimeout(resolvePromise, ms)
  })
}

main().catch((error) => {
  console.error('[listUnannotated] Failed.')
  console.error(error)
  process.exitCode = 1
})
