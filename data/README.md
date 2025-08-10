# Data scripts

**en_et_and_ru_et** folder contains files related to the preparation of the combined English-Estonian and Russian-Estonian fine-tuning dataset, including partial sequences (proportion from each audio sample and corresponding transcription and translation text). Output format is suitable for Seamless fine-tuning.

**et_en_and_et_ru** folder contains files for preparing the combined Estonian-English and Estonian-Russian dataset for fine-tuning. Includes partial sequences and output format is also suitable for Seamless fine-tuning. In addition, audio files that were not originally 16000 Hz, were resampled and saved in .wav format.

**evaluation** folder contains the scripts for preparing FLEURS test split for evaluation with SimulEval.

The `combine.py` is can be used to combine the above mentioned datasets into one combine dataset. It samples an equal amount from each direction (Estonian-English, Estonian-Russian, English-Estonian and Russian-Estonian) and splits the chosen samples 95/5 into train and eval splits. The eval split is not used for final evaluation, but only for early stopping in the training.

`metadata.py` is not really important, but just for calculating the metadata associated with a 
specific finetuning file. Useful for creating tables.