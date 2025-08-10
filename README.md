# Estonian Simultaneous Speech-to-Text Machine Translation

This repository contains files related to the Estonian simultaneous S2TT project.  
The translation directions are Estonian-English, Estonian-Russian, English-Estonian and Russian-Estonian.

**Agents** folder contains SimulEval agent files together with scripts that can be used to run evaluation. Parameters like source and target are standard SimulEval parameters and more info about them is in the SimulEval project. Running them most likely requires some reading about the SimulEval project first.

**Metrics** folder containts the chrF++ implementation for SimulEval. The translation quality was evaluated using BLEU, which SimulEval supports out of the box, and chrF++. The metric must be added to the SimulEval `quality_scorer.py`file. Then it can be used by adding `--quality-metrics chrf` to the `simuleval` command or by calculating scores on the previous outputs using `simuleval --score-only --quality-metrics chrf --output output_folder`.

**Data** folder contains data processing related scripts. Some of the scripts are specific to the server, where the files were stored, so not applicable to anywhere else, but most of the code can likely be reused.

**Finetuning** folder contains the scripts that were used for finetuning NLLB-200 distilled 1.3B and Seamless M4Tv2 large models. Works using data prepared with the scripts in the `data` folder.
