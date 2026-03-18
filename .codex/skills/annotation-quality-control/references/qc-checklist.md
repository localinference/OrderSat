# QC Checklist

Use this checklist when auditing one file or a batch.

## 1. Pair The Files

Find the input text that matches the output file name.

Use `./cli/02_process_training_samples/getInputPathFromOutputPath/index.js` or the same replacement rule:

- `/outputs/` -> `/inputs/`
- `.jsonld` -> `.txt`

Do not review an output without its source OCR text.

## 2. Confirm The Document Type

Ask:

- Is this fundamentally a receipt, order confirmation, invoice-like purchase record, pickup notice, or shipment notice?
- If yes, is the main object still appropriately modeled as `Order`?

Do not force a different top-level type unless the current type is clearly wrong.

## 3. Check Grounding

For each important field, ask whether the OCR text actually supports it.

High-priority fields:

- merchant name
- order number
- date and time
- line items
- subtotal
- taxes
- fees
- grand total
- payment method or payment status
- delivery or pickup details

If a fact is not grounded, remove or correct it.

## 4. Check Structure

Known fields must stay in known Schema.org properties.

Examples:

- phone in `telephone`
- merchant name in `name`
- order id in `orderNumber`
- street line in `streetAddress`

Do not move damaged known fields into `additionalProperty`. Keep the structure and mark the value if needed.

Treat these as structural mistakes:

- `PostalAddress` used for room labels, counters, or seating labels
- `additionalProperty` used as a dumping ground for known receipt fields
- meta-review notes stored as receipt data

## 5. Check Error Marking

Re-read the annotation skill if the distinction is fuzzy.

Use `[ERROR:UNCERTAIN]` when:

- the field is known
- the value looks suspicious
- it could still be legitimate

Use `[ERROR:FATAL]` when:

- the field is known
- exact machine meaning matters
- corruption makes safe repair impossible

Typical fatal cases:

- phone numbers
- order numbers
- transaction ids
- emails
- URLs
- tracking numbers
- exact dates or times when a wrong character changes meaning

Do not "clean up" the marked raw OCR fragment after the prefix.

## 6. Check Arithmetic

Recompute the money trail from the OCR when the output contains:

- subtotal
- tax
- service charge
- discounts
- total
- cash tendered
- change due

Treat arithmetic corrections as confirmed only when the receipt clearly supports the corrected value.

Examples:

- subtotal plus tax equals printed total
- tendered amount minus change due reveals the payable total
- taxable amount reveals a misread service charge

If multiple arithmetic interpretations remain plausible because the OCR is too damaged, do not force a correction.

## 7. Distinguish Real Mistakes From Alternative Valid Annotations

Fix:

- contradictions
- inventions
- broken structure
- wrong math
- clearly wrong normalization

Do not fix:

- stylistic differences
- sparse modeling that is still valid
- ambiguous OCR with no decisive reading
- a conservative omission that follows the annotation rules

## 8. Revalidate Touched Files

Run:

```powershell
node cli/validateStructure "./src/02_training_samples/outputs/<requestedLanguage>/<sample>.jsonld"
```

If validation fails after a change, fix the structure before finishing.

## 9. Report Findings

When the task is framed as a review, report findings before summaries.

For each finding, include:

- the output file
- the specific issue
- why it is a mistake
- the correction made, if any

If no confirmed mistakes are found, say that explicitly and note any residual ambiguity that was left unchanged.
