const cache = new Map()

export async function loadLanguageConfig(language) {
  if (cache.has(language)) {
    return cache.get(language)
  }

  const [labelsModule, valuesModule] = await Promise.all([
    import(new URL(`../${language}/labels/index.js`, import.meta.url)),
    import(new URL(`../${language}/values/index.js`, import.meta.url)),
  ])

  const config = {
    language,
    labels: labelsModule.default,
    values: valuesModule.default,
  }

  cache.set(language, config)
  return config
}
