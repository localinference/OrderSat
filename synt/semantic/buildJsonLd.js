import { toMoneyString } from '../values/factories.js'

export function buildJsonLd(record) {
  const graph = []

  graph.push({
    '@id': '#seller',
    '@type': 'Organization',
    name: mark(record, 'seller.name', record.seller.name),
    telephone: mark(record, 'seller.telephone', record.seller.telephone),
    email: mark(record, 'seller.email', record.seller.email),
    taxID: record.seller.taxId,
    url: record.seller.website,
    address: buildPostalAddress(record, 'seller.address', record.seller.address),
  })

  graph.push({
    '@id': '#customer',
    '@type': record.customer.type,
    name: mark(record, 'customer.name', record.customer.name),
    email: record.customer.email,
    identifier: {
      '@type': 'PropertyValue',
      name: 'Customer ID',
      value: record.customer.identifier,
    },
    address: buildPostalAddress(
      record,
      'customer.address',
      record.customer.address
    ),
  })

  if (record.delivery) {
    graph.push({
      '@id': '#shipper',
      '@type': 'Organization',
      name: record.delivery.providerName,
    })

    graph.push({
      '@id': '#delivery',
      '@type': 'ParcelDelivery',
      deliveryAddress: buildPostalAddress(
        record,
        'delivery.address',
        record.delivery.address
      ),
      provider: { '@id': '#shipper' },
    })
  }

  for (const [index, item] of record.items.entries()) {
    graph.push({
      '@id': `#offer-${index + 1}`,
      '@type': 'Offer',
      price: toMoneyString(item.linePrice),
      priceCurrency: record.currencyCode,
      eligibleQuantity: {
        '@type': 'QuantitativeValue',
        value: item.quantity,
        unitText: item.unitText,
      },
      itemOffered: { '@id': `#item-${index + 1}` },
      priceSpecification: {
        '@type': 'UnitPriceSpecification',
        price: toMoneyString(item.unitPrice),
        priceCurrency: record.currencyCode,
        referenceQuantity: {
          '@type': 'QuantitativeValue',
          value: 1,
          unitText: item.unitText,
        },
      },
    })

    graph.push({
      '@id': `#item-${index + 1}`,
      '@type': item.type,
      name: mark(record, `items.${index}.name`, item.name),
      sku: item.sku,
    })
  }

  const additionalProperty = [
    propertyValue(
      'Subtotal',
      `${record.currencyCode} ${toMoneyString(record.amounts.subtotal)}`
    ),
    propertyValue(
      'Tax Amount',
      `${record.currencyCode} ${toMoneyString(record.amounts.taxAmount)}`
    ),
    propertyValue('Receipt Type', record.order.receiptType),
    propertyValue('Service Mode', record.order.serviceMode),
    propertyValue('Cashier Name', record.order.cashierName),
    propertyValue('Register Number', record.order.registerNumber),
  ]

  if (record.amounts.discountAmount > 0 && record.order.promoCode) {
    additionalProperty.push(
      propertyValue('Discount Code', record.order.promoCode),
      propertyValue(
        'Discount Amount',
        `${record.currencyCode} ${toMoneyString(record.amounts.discountAmount)}`
      )
    )
  }

  if (record.delivery?.trackingNumber) {
    additionalProperty.push(
      propertyValue(
        'Tracking Number',
        mark(record, 'delivery.trackingNumber', record.delivery.trackingNumber)
      )
    )
  }

  if (record.delivery?.shippedDateIso) {
    additionalProperty.push(propertyValue('Shipped Date', record.delivery.shippedDateIso))
  }

  graph.push({
    '@id': '#order',
    '@type': 'Order',
    orderNumber: mark(record, 'order.number', record.order.number),
    orderDate: record.order.dateIso,
    orderStatus: record.order.orderStatusUrl,
    paymentMethod: record.order.paymentMethodUrl,
    paymentStatus: record.order.paymentStatusUrl,
    seller: { '@id': '#seller' },
    customer: { '@id': '#customer' },
    acceptedOffer: record.items.map((_item, index) => ({
      '@id': `#offer-${index + 1}`,
    })),
    totalPaymentDue: {
      '@type': 'PriceSpecification',
      price: toMoneyString(record.amounts.total),
      priceCurrency: record.currencyCode,
    },
    additionalProperty,
    ...(record.delivery ? { orderDelivery: { '@id': '#delivery' } } : {}),
  })

  return {
    '@context': 'https://schema.org',
    '@graph': graph,
  }
}

function buildPostalAddress(record, basePath, address) {
  return {
    '@type': 'PostalAddress',
    streetAddress: mark(
      record,
      `${basePath}.streetAddress`,
      address.streetAddress
    ),
    addressLocality: address.addressLocality,
    addressRegion: address.addressRegion,
    postalCode: address.postalCode,
    addressCountry: address.addressCountry,
  }
}

function propertyValue(name, value) {
  return {
    '@type': 'PropertyValue',
    name,
    value,
  }
}

function mark(record, path, value) {
  const errorClass = record.errors[path]
  if (!errorClass) {
    return value
  }
  return `[ERROR:${errorClass}] ${value}`
}
