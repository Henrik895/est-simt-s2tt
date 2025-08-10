# Preparing FLEURS test split for evaluation

Prepares the FLEURS test split for evaluation on the Estonian-English, Estonian-Russian, English-Estonian, English-Russian, Russian-Estonian and Russian-English directions.

`load_data.py` loads the FLEURS data and `prep_data.py` creates the structure suitable for running the SimulEval evaluations.

The FLEURS data has transcriptions and raw_transcriptions. In this case, raw_transcriptions were used because they include upper and lower cases, punctuations etc.
