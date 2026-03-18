import Validator from '@adobe/structured-data-validator'
// Fetch the current schema.org schema
const schemaOrgJson = await (
  await fetch('https://schema.org/version/latest/schemaorg-all-https.jsonld')
).json()
// Create a validator instance singleton
const validator = new Validator(schemaOrgJson)

export async function validateStructure(output) {
  // Validate the model produced structured data
  const issues = await validator.validate(output)
  if (issues.length === 0) return true
  else return issues
}
