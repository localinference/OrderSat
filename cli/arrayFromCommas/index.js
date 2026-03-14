export function arrayFromCommas(value) {
  return value.split(',').map((item) => item.trim())
}
