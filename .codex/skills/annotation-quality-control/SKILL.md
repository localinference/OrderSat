---
name: annotation-quality-control
description: Review existing training-sample annotations in `./src/02_training_samples/outputs/{language}/*.jsonld` against their source OCR text in `./src/02_training_samples/inputs/{language}/*.txt`, check whether the JSON-LD is grounded and consistent with the annotation policy, and fix only confirmed annotation mistakes. Use when Codex must audit receipt or order-sample annotations, perform self-QC after annotation work, or validate whether a suspected issue is a real mistake versus an ambiguous OCR case that should remain unchanged.
---

# Annotation Quality Control

## Goal

Audit existing annotated receipt and order samples without re-annotating them from scratch.

Find and fix only mistakes that are clearly attributable to the annotation, not ambiguity in the source OCR.

Read [../annotate-input-sample-to-ouput-sample/SKILL.md](../annotate-input-sample-to-ouput-sample/SKILL.md) before making QC decisions. Read [references/qc-checklist.md](references/qc-checklist.md) when reviewing batches or when a case is borderline.

## Scope

Read:

- `./src/02_training_samples/outputs/<requestedLanguage>/*.jsonld`
- `./src/02_training_samples/inputs/<requestedLanguage>/*.txt`

Map each output back to its input with:

- `./cli/02_process_training_samples/getInputPathFromOutputPath/index.js`

Write:

- corrected versions of the same `.jsonld` files only when there is a confirmed mistake

Do not create replacement outputs for samples that do not already exist unless the user explicitly asks for annotation work instead of QC.

## Review Standard

Treat the existing output as presumptively valid until the input text proves otherwise.

Prefer leaving an ambiguous value unchanged over forcing a speculative correction.

Fix only when at least one of these is true:

- the output invents a fact not supported by the input
- the output contradicts a clearly printed value
- the output breaks the annotation skill's structure rules
- the output uses the wrong error class for a clearly resolvable case
- the output performs arithmetic or normalization incorrectly
- the output contains commentary, meta-notes, or reviewer text instead of receipt data
- the output fails structural validation

If the OCR itself is inconsistent and no single correction is clearly defensible, leave the value as-is or preserve the existing error marking.

## Workflow

1. Identify the target outputs the user wants reviewed.
2. For each output, read the corresponding input text.
3. Compare the JSON-LD against the OCR line by line for grounded facts, structure, and arithmetic.
4. Check the sample against the annotation skill's modeling and error-marking rules.
5. Edit only the files with confirmed annotation mistakes.
6. Revalidate every touched file with `node cli/validateStructure`.
7. Report findings first, ordered by severity, then summarize fixes made.

## What To Check

Check whether the output:

- models the document as `Order` when the document is fundamentally a receipt or order record
- keeps known facts in native Schema.org fields instead of falling back to `additionalProperty`
- uses `[ERROR:UNCERTAIN]` only for suspicious-but-possibly-legitimate values
- uses `[ERROR:FATAL]` only for materially broken exact identifiers or machine-critical values
- omits unsupported fields instead of inventing them
- preserves seller, delivery, payment, totals, and item structure correctly
- copies exact values faithfully when the input is clear
- normalizes OCR noise only when the local evidence strongly supports the normalized value
- keeps monetary math internally consistent
- avoids reviewer commentary or process notes inside the JSON-LD

## Editing Rule

Make the smallest defensible correction.

Do not rewrite a whole sample just because you would annotate it differently today.

Do not downgrade a sparse but valid annotation into a richer speculative one.

When fixing a mistake, preserve all unaffected grounded facts and existing ids where possible.

## Mistake Categories

Common confirmed mistakes include:

- invented merchant, customer, address, email, URL, phone, or identifier values
- incorrect totals, taxes, subtotals, service charges, tendered amounts, or change due
- placing a known field into `additionalProperty` instead of its native property
- using address structure for non-address labels
- copying OCR labels as receipt facts when they are only layout noise
- putting meta commentary like "missing date" or "unclear value" into the training output
- silently repairing unique identifiers that should have been left marked or omitted

Common non-mistakes include:

- ugly but plausible merchant or product names
- OCR ambiguity where multiple readings are possible
- sparse outputs that omit low-confidence facts
- unresolved values already marked correctly under the annotation skill

## Validation

After every edit, run:

```powershell
node cli/validateStructure "./src/02_training_samples/outputs/<requestedLanguage>/<sample>.jsonld"
```

Treat validation as necessary but not sufficient. A file can be structurally valid and still be semantically wrong.

## Final Checklist

Before finishing, verify:

- every changed file has a confirmed reason for the change
- no unchanged ambiguous case was "fixed" speculatively
- native Schema.org structure is preserved
- error markers still match the annotation skill
- arithmetic corrections are supported by the document
- every touched file passes `node cli/validateStructure`
