simuleval \
    --agent end-to-end.py \
    --wait-k 2 \
    --source-segment-size 1000 \
    --source en_et/source.txt \
    --source-language eng \
    --target en_et/target.txt \
    --target-language est \
    --output output_end_to_end/en_et \
    --device cuda:0 \
    --silence-limit-ms 500

simuleval \
    --agent end-to-end.py \
    --wait-k 2 \
    --source-segment-size 1000 \
    --source en_ru/source.txt \
    --source-language eng \
    --target en_ru/target.txt \
    --target-language rus \
    --output output_end_to_end/en_ru \
    --device cuda:0 \
    --silence-limit-ms 500

simuleval \
    --agent end-to-end.py \
    --wait-k 2 \
    --source-segment-size 1000 \
    --source et_en/source.txt \
    --source-language est \
    --target et_en/target.txt \
    --target-language eng \
    --output output_end_to_end/et_en \
    --device cuda:0 \
    --silence-limit-ms 500

simuleval \
    --agent end-to-end.py \
    --wait-k 2 \
    --source-segment-size 1000 \
    --source et_ru/source.txt \
    --source-language est \
    --target et_ru/target.txt \
    --target-language rus \
    --output output_end_to_end/et_ru \
    --device cuda:0 \
    --silence-limit-ms 500

simuleval \
    --agent end-to-end.py \
    --wait-k 2 \
    --source-segment-size 1000 \
    --source ru_en/source.txt \
    --source-language rus \
    --target ru_en/target.txt \
    --target-language eng \
    --output output_end_to_end/ru_en \
    --device cuda:0 \
    --silence-limit-ms 500

simuleval \
    --agent end-to-end.py \
    --wait-k 2 \
    --source-segment-size 1000 \
    --source ru_et/source.txt \
    --source-language rus \
    --target ru_et/target.txt \
    --target-language est \
    --output output_end_to_end/ru_et \
    --device cuda:0 \
    --silence-limit-ms 500
