const list = [
  './src/02_training_samples/outputs/eng/-29ToZ2-VWLA86e-qN54LYqG-s8TmcEt2imXnVjJoa28NzHCBZY7ZpRCbIQ4ifMC.jsonld',
  './src/02_training_samples/outputs/eng/-2xpd6OgsEKeVE-jkgh23ykNd61ldLwyIOct-YRWklE0-88_x0AGWFxOxVKIY23p.jsonld',
  './src/02_training_samples/outputs/eng/-3XOLkD8vWzCs7u7DKMZbEIc-BvSEzjqLuA6urbtcMMAI0UEvuxdbpuhKiPCI12n.jsonld',
  './src/02_training_samples/outputs/eng/-47n3urhcO-h-fxrJdb71kX_nAKYUBv3SddiEvoiXxzHQfOCU0xOh1fuU1fwDRG4.jsonld',
  './src/02_training_samples/outputs/eng/-4bzi6i4VvDCJqSFj0LisKAwikY7HC_KbAjTZVgtQ6WNfH0Z3U0FWDkoQHat5hMN.jsonld',
  './src/02_training_samples/outputs/eng/-6GFKfT3x1ScKyp1pTC_RBuD5BOHB3Yh_kr9b2DenXlAIUb-4wpaEKdKIPXNu1Zi.jsonld',
  './src/02_training_samples/outputs/eng/-6KgGv5W4ipoaw_Q_GnlNf2S-vreLItxMTpSyE9IasnkBY60PK-GmulOykSOG3Mm.jsonld',
  './src/02_training_samples/outputs/eng/-7rwcsFSFpfpwa5ia-4_N89wNYD9swd6U3hlljDiEs4UDm2gepwnYrYVV3DVJtQN.jsonld',
  './src/02_training_samples/outputs/eng/-a2u-Rdv218xHzZ7FzWmHV_R4oa1G6b5ViGymdl9SKWutHUjpKsmvJltUDrd8p7C.jsonld',
  './src/02_training_samples/outputs/eng/-A3uovV0g8hZvYxzW6w15hzAhBjdfC38YkPr4jPGdvNjrWIdbsrOIj6F9y9j8dTl.jsonld',
  './src/02_training_samples/outputs/eng/_-ygtgqAyvb-4WN2dwtIyA0QGGPv8BmxVLVjdTunIdSrDWrkITMN3FztYUumGkzi.jsonld',
  './src/02_training_samples/outputs/eng/_0AUwFUxNq4Z1PIqJpL3dHLtjAO6Wl-RqhrT1IJmQeGrPoVbGq2QLxft7PxJP_wi.jsonld',
  './src/02_training_samples/outputs/eng/_0wIjaXHa0QklPl-LIJw0dITbUPU-vCZ-_xbZGWBKU-LKILTfCouCpsPIX6EDLa_.jsonld',
  './src/02_training_samples/outputs/eng/_23rZCfNpgWWQgW2HWMJ0Oy3beYtsuYTvG2MGPKBQXwEsaQ2-ebIuY8lnADElNVb.jsonld',
  './src/02_training_samples/outputs/eng/_5lJu5d6VJ4_yc8lBwKDW1Q3aJkCiW_MMlQ91hdoo4YH5vJCnVnINuXSnrCuomUq.jsonld',
  './src/02_training_samples/outputs/eng/_6f1RzXYxHztYq-aHkkvtm7vTAfmlozswjeV4Mf0-TuJVWGJ81Sv4vm21NPaFdgR.jsonld',
  './src/02_training_samples/outputs/eng/_6h1GPvdhuTZDp14bJGazXhtN8U3W8p_6bU3IsCZK0eaaJAK7Edk7IBSZC5_nnRd.jsonld',
  './src/02_training_samples/outputs/eng/_6pVCSqPv9_pUGCHX62viKawxK1nnXSEZhwDNB7UvjqczXEYLyMBYJGNI1ad5T90.jsonld',
  './src/02_training_samples/outputs/eng/_7OhoFDw49jfzseyje2aDR0O4PJpoHdRwRNx3-aSEwDr8dLtvjc6rwhr1-0XroBf.jsonld',
  './src/02_training_samples/outputs/eng/_8Tg2ghF6a-hzcDmUk-lHHy4JMaNpyz4EW7Ld-_qua1AcKbDhCbmA5vbJVZ_AYHI.jsonld',
]

export function getOutputPathsFromInputPaths(outputPaths) {
  const inputPaths = [...outputPaths].map((path) => {
    path.replace('outputs', 'inputs')
    path.replace('jsonld', 'txt')
  })
  return inputPaths
}
