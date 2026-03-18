📘 Guideline: epochs, early stopping ja min_delta datasetin koon mukaan

Huom: DELTA tarkoittaa pienintä validointitappion parannusta, joka lasketaan oikeaksi edistykseksi.

Dataset < 1 000 samplea (pieni dataset)

Epochs: 80–120

Early stopping patience: 15–25

Min_delta: 1e-5 – 1e-4

Perustelu: malli tarvitsee monta kierrosta nähdäkseen data-varianssin; validation loss epävakaa → pieni delta, jotta pienet parannukset tunnistetaan

Dataset 10 000 – 1 000 000 samplea (keskikokoinen dataset)

Epochs: 15–30

Early stopping patience: 5–10

Min_delta: 1e-4 – 1e-3

Perustelu: yksi epoch kattaa jo paljon variaatiota; delta hieman suurempi, jotta satunnaisheilahtelut eivät laukaise turhaa early stoppingia

Dataset > 1 000 000 samplea (iso dataset)

Epochs: 3–5

Early stopping patience: 0.5–2

Min_delta: 1e-3 – 1e-2

Perustelu: dataset on niin suuri, että pienet muutokset validoinnissa eivät ole merkityksellisiä; early stopping voi olla step-pohjainen

🔧 Käytännön sovellus
if sample_count < 1000:
epochs = 100
patience = 20
min_delta = 1e-5
elif sample_count < 1_000_000:
epochs = 20
patience = 5
min_delta = 1e-4
else:
epochs = 5
patience = 1
min_delta = 1e-3
💡 Huomioita

Nämä ovat default-arvoja, aina tarkista validointilossin käyttäytyminen.

Pienillä dataseteillä kannattaa käyttää pientä batch sizea + gradient accumulationia.

Isoilla dataseteillä early stopping kannattaa yhdistää step-pohjaiseen seurantaan.

Min_delta yhdessä patience:n kanssa ohjaa, milloin training lopetetaan — ei pelkkä epoch-luku riitä.
