---
name: annotate-input-sample-to-ouput-sample
description: Convert OCR-extracted receipt and order-confirmation text in `./src/02_training_samples/inputs/{language}/*.txt` into conservative Schema.org JSON-LD in `./src/02_training_samples/outputs/{language}/{input}.jsonld`. Use when Codex must produce order-centered training samples, keep Schema.org structure explicit, normalize grounded OCR noise, and annotate unresolved native-field values with `[ERROR:UNCERTAIN]` or `[ERROR:FATAL]` instead of guessing.
---

# Annotate Input Sample To Output Sample

## Goal

Read one OCR-extracted receipt or order-related text file and write one corrected, normalized JSON-LD sample.

Model the document as `Order` by default whenever it is fundamentally a receipt, purchase confirmation, checkout confirmation, pickup notice, shipped-order summary, service purchase confirmation, or other transaction record.

Prefer sparse, correct JSON-LD over rich guessed JSON-LD.

Read [references/error-marking.md](references/error-marking.md) when deciding between `[ERROR:UNCERTAIN]` and `[ERROR:FATAL]` or when placing marked values.

## Input And Output

Before selecting the next sample, sync the unannotated working queue with:

```powershell
node cli/listUnannotated --language <requestedLanguage>
```

Read:

- `./temp-unannotated/<requestedLanguage>/*.txt` by default
- `./src/02_training_samples/inputs/<requestedLanguage>/*.txt` only when the user explicitly points to a specific source file

Treat `./src/02_training_samples/inputs` and `./src/02_training_samples/outputs` as the source of truth.
Treat `./temp-unannotated` as a regenerated working queue only.

Write:

- `./src/02_training_samples/outputs/<requestedLanguage>/<inputSampleFileName>.jsonld`

After writing the output sample, delete the matching working-queue file:

- `./temp-unannotated/<requestedLanguage>/<inputSampleFileName>.txt`

If the input language does not match the language implied by the directory, ignore that sample.

If the user asks for the "next" samples, take them from `./temp-unannotated/<requestedLanguage>` in lexicographic order unless the user specifies another ordering rule.

## Output Contract

Produce:

- valid JSON
- valid JSON-LD
- only the JSON-LD object
- no markdown fences
- no explanations
- no citations
- no debug notes

Use:

- `"@context": "https://schema.org"`
- a single top-level object or `"@graph"` when multiple linked entities are needed
- stable local ids such as `"#order"`, `"#merchant"`, `"#customer"`, `"#offer-1"`, `"#product-1"`

## Default Modeling

Use `Order` as the primary type unless the document is clearly not a transaction record.

Use these types when justified by the evidence:

- `Order`
- `OrderItem`
- `Offer`
- `Product`
- `Service`
- `Organization`
- `LocalBusiness`
- `Store`
- `Person`
- `PostalAddress`
- `QuantitativeValue`
- `PriceSpecification`
- `PropertyValue`
- `ParcelDelivery`

Prefer:

- `seller` for the merchant
- `acceptedOffer` for purchased lines
- `itemOffered` as `Product` or `Service`
- `orderDelivery` for shipping or pickup data
- `additionalProperty` only for real receipt facts that do not have a better Schema.org property

## Core Rules

Ground every important value in the receipt text itself.

Normalize only when the correction is strongly supported by the local document evidence.

Do not invent:

- merchant names
- phone numbers
- company numbers
- order numbers
- transaction ids
- customer identities
- addresses
- URLs
- emails
- barcode digits

Prefer omission over fabrication.

## Native-Field Structure Rule

If the field identity is known, keep the proper Schema.org structure and put the error marker in the native field value.

This rule is mandatory.

Examples:

- known phone field: use `telephone`, not `additionalProperty`
- known merchant name: use `name`, not `additionalProperty`
- known order id: use `orderNumber`, not `additionalProperty`
- known street line: use `streetAddress`, not `additionalProperty`

`additionalProperty` is only for facts that truly have no better native property. It is never a fallback for damaged values in otherwise known fields.

## Decision Rule

For every noisy value, choose exactly one path:

1. Normalize it when the intended value is strongly supported by the receipt.
2. Mark it with `[ERROR:UNCERTAIN]` when the field is identifiable but the suspicious value could still be a legitimate name, label, spelling, or brand form.
3. Mark it with `[ERROR:FATAL]` when the field is identifiable but the exact value is materially broken and any repair would be guesswork.
4. Omit it when even the field identity is unclear.

Never emit both error classes for the same value.

## Error System

Use exactly these two prefixes:

- `[ERROR:UNCERTAIN]`
- `[ERROR:FATAL]`

Put the prefix at the start of the raw OCR value kept in the native field.

Examples:

- `"name": "[ERROR:UNCERTAIN] Kaarinan Herkkuu"`
- `"telephone": "[ERROR:FATAL] 04O-12A-77B"`
- `"orderNumber": "[ERROR:FATAL] AB1Z7O4"`

Do not silently clean marked values. Preserve the raw OCR fragment after the prefix.

### `[ERROR:UNCERTAIN]`

Use `[ERROR:UNCERTAIN]` when all of the following are true:

- the field identity is reasonably clear
- the value looks suspicious
- the value could still be legitimate
- calling it wrong would require knowledge the receipt does not prove

Typical cases:

- merchant names
- person names
- product names
- street names
- free-text labels
- stylized branding

### `[ERROR:FATAL]`

Use `[ERROR:FATAL]` when all of the following are true:

- the field identity is clear
- the exact value matters materially
- the corruption breaks machine usefulness
- any repair would be a guess

Typical cases:

- phone numbers
- emails
- URLs
- order numbers
- receipt numbers
- VAT or tax ids
- loyalty ids
- transaction ids
- tracking numbers
- card tail digits when unclear
- exact dates, times, totals, or numeric identifiers when a wrong character changes the meaning

## Native-Field Examples

Known field plus uncertain name:

```json
{
  "@id": "#merchant",
  "@type": "Store",
  "name": "[ERROR:UNCERTAIN] Kaarinan Herkkuu"
}
```

Known field plus fatal phone:

```json
{
  "@id": "#merchant",
  "@type": "Store",
  "telephone": "[ERROR:FATAL] 04O-12A-77B"
}
```

Known structured address plus uncertain street:

```json
{
  "@type": "PostalAddress",
  "streetAddress": "[ERROR:UNCERTAIN] Mannerheirnintie 5"
}
```

Known order id plus fatal corruption:

```json
{
  "@id": "#order",
  "@type": "Order",
  "orderNumber": "[ERROR:FATAL] AB1Z7O4"
}
```

## Normalization Rule

Normalize only when the receipt strongly supports the intended reading.

Examples of acceptable normalization when grounded:

- `T0TAL` to `TOTAL`
- `CASHH` to `CASH`
- `2O24-0I-3I` to `2024-01-31`
- `EUP` to `EUR` when the surrounding receipt clearly indicates euro

Do not use this logic to repair unique identifiers.

## Omission Rule

Omit the field entirely when:

- the field identity is unclear
- the OCR fragment could map to multiple meanings
- the correction would require guessing
- the document does not actually support the fact

Do not preserve garbage just to fill structure.

## Common Modeling Guidance

Use the merchant as `seller`.

Represent purchased lines as `Offer` objects, optionally with `OrderItem` when that genuinely helps.

Use `Product` or `Service` for the purchased thing.

Use `PostalAddress` only when the printed address structure is actually supported.

Use `ParcelDelivery` only when the document includes delivery or pickup information tied to the order.

Use `paymentStatus: "https://schema.org/PaymentComplete"` only when the document clearly indicates completed payment.

## Validation

Validate the finished sample with:

```powershell
node cli/validateStructure "./src/02_training_samples/outputs/<requestedLanguage>/<inputSampleFileName>.jsonld"
```

Treat `true` as structurally valid. If the validator returns issues, fix the JSON-LD shape. Error markers annotate values; they do not excuse invalid structure.

## Final Checklist

Before finishing, verify:

- the document is modeled primarily as `Order` when appropriate
- every important value is grounded in the input
- native Schema.org fields are preserved whenever field identity is known
- general OCR noise was normalized only when strongly supported
- unique or exact values were never guessed
- unresolved values use either `[ERROR:UNCERTAIN]` or `[ERROR:FATAL]`, not both
- unclear fields were omitted
- the result is valid JSON-LD
