# Error Marking Reference

## Principle

Keep structure and value handling separate.

If the field identity is known, emit the correct Schema.org field and place the marker inside that field value.

Do not move a known field into `additionalProperty` just because the value is damaged.

## Marker Classes

Use `[ERROR:UNCERTAIN]` when the value may simply be a legitimate name, label, spelling, or brand variant.

Use `[ERROR:FATAL]` when the field is known but the exact value is machine-critical and OCR corruption makes safe recovery impossible.

## Native-Field Defaults

Use `[ERROR:UNCERTAIN]` by default for these when suspicious but still plausibly legitimate:

- `name`
- `streetAddress`
- `description`
- `itemOffered.name`
- other human-language labels

Use `[ERROR:FATAL]` by default for these when corrupted:

- `telephone`
- `email`
- `url`
- `orderNumber`
- `confirmationNumber`
- `trackingNumber`
- `paymentMethodId`
- exact date or time fields
- exact totals and other numeric identifiers

## Examples

Merchant name:

```json
{
  "@type": "Store",
  "name": "[ERROR:UNCERTAIN] Kaarinan Herkkuu"
}
```

Merchant phone:

```json
{
  "@type": "Store",
  "telephone": "[ERROR:FATAL] 04O-12A-77B"
}
```

Order number:

```json
{
  "@type": "Order",
  "orderNumber": "[ERROR:FATAL] AB1Z7O4"
}
```

Street line:

```json
{
  "@type": "PostalAddress",
  "streetAddress": "[ERROR:UNCERTAIN] Mannerheirnintie 5"
}
```

## `additionalProperty`

Use `additionalProperty` only for receipt facts that truly have no better Schema.org property.

Valid examples:

- terminal id when no better property is available
- lane number
- cashier code
- internal store notice code

Invalid use:

- putting `telephone` into `additionalProperty`
- putting `name` into `additionalProperty`
- putting `orderNumber` into `additionalProperty`
- putting `streetAddress` into `additionalProperty`
