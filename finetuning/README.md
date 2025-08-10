# Fine-tuning scripts for NLLB-200 and Seamless M4Tv2 large

`run_finetune_m4t.sh` script is used for fine-tuning Seamless M4Tv2. It is based on the default fine-tuning script provided by Seamless project, but with some changed parameters.

`run_finetune_nllb.sh` uses `finetune_nllb.py` to fine-tune a NLLB distilled 1.3B model. Because NLLB expects a different training data format compared to the Seamless, the `finetune_nllb.py` also processes the dataset first before proceeding to the training loop.
