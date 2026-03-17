import { Validator } from '@adobe/structured-data-validator'
// Fetch the current schema.org schema
const schemaOrgJson = await (
  await fetch('https://schema.org/version/latest/schemaorg-all-https.jsonld')
).json()
// Create a validator instance singleton
const validator = new Validator(schemaOrgJson)
//
export async function validateModelOuput(output) {
  // Validate the model produced structured data
  return await validator.validate(output)
}
