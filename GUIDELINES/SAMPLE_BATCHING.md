1. BATCH_SIZE

Mikä se on: Kuinka monta samplea syötetään mallille kerralla ennen weight updatea.

Suositeltu alue:

Pienet datasetit / rajoitettu VRAM: 1–4

Keskikokoiset datasetit: 8–16

Suuret datasetit / riittävästi muistia: 32–64+

Miksi:

Pieni batch → gradientti kohinainen → voi auttaa generalizationissa (flat minima)

Liian pieni batch → oppiminen epävakaata, training hidas

Iso batch → stabiilinen gradientti, nopeampi training, mutta riski huonommasta yleistymisestä

Milloin käyttää:

Käytä pientä batchia, kun VRAM rajoittaa ja haluat parantaa yleistymistä pienillä dataseteillä.

2. ACCUMULATION_STEPS

Mikä se on: Kuinka monta batchia lasketaan ennen painojen päivitystä. Efektiivinen batch size = BATCH_SIZE \* ACCUMULATION_STEPS.

Suositeltu alue:

Pieni BATCH_SIZE → accumulation 8–32, riippuen VRAMista

Suuri BATCH_SIZE → accumulation usein 1

Miksi:

Säästää VRAMia, mutta antaa efektiivisen suuren batchin hyödyt: stabiilinen gradientti, parempi throughput

Pienentää gradientin kohinaa ilman että tarvitsee kasvattaa fyysistä batchia

🔧 Käytännön esimerkki

# Pieni dataset, vähän VRAM

BATCH_SIZE = 1
ACCUMULATION_STEPS = 16 # Efektiivinen batch = 16

# Keskikokoinen dataset

BATCH_SIZE = 8
ACCUMULATION_STEPS = 4 # Efektiivinen batch = 32

# Iso dataset, riittävästi muistia

BATCH_SIZE = 32
ACCUMULATION_STEPS = 1 # Efektiivinen batch = 32
💡 Yhteenveto

BATCH_SIZE määrää kuinka monta samplea nähdään kerralla

ACCUMULATION_STEPS kompensoi pienen batchin pienemmän muistin vuoksi, mutta mahdollistaa stabiilin oppimisen

Yhdessä ne määrittävät efektiivisen batchin, joka vaikuttaa learning rateen, gradientin vakauteen ja trainingin nopeuteen
