## HIGH-LEVEL GOALS

We want to teach the model to generalize on transforming [order like arbitrary text](../WHAT_IS_AN_ORDER.md) in to a schema.org/Order, that should be taken in to account when desinging the synthtization algorithms.

We want the model to encode the data in to a semantic presentation then move the semantic pieces to the semantics mathcing schema org field

## POSSIBLE FLOW

read config per language (added per language possible labels and values or sets of data that can be combined in to data) pick randomly a set of fields and values build a input presentation plain text with artificially created random ocr noise, clean plaintext, html, xml,csv, json and other possible structured text formats and a output "annotated" (generated correctly) schema.org/Order JSON-LD hash the input pair see src/02_training/samples/inputs/${language}/${SHA-384}.txt if exists continue to next iter, if doesn't exist write input to src/02_training/samples/inputs/${language}/${SHA-384}.txt output to src/02_training/samples/outputs/${language}/${SHA-384}.jsonld continue until all possible combinations that make sense for goals are iterated on, do this for every langauage console.log how many were written and how long it took per language

see C:\Users\jorts\OrderSaT\synthesizer\config  
it can be desinged to something entirely different the key is that per language vocabulary that is needed to generate a enormous/huge/massive ammount of receipts can be added cleanly, in a clear structured managed flow there should be variance in labels there should be data without an label everything the goal is to try and cover as much unique order like stuff ass possible, as many ocr noises as possible but do it with max performance maybe a runWithWorker style thing where we utilize a worker pool for concurrency see C:\Users\jorts\OrderSaT\cli\01_process_data_sources
