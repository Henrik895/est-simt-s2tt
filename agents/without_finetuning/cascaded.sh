simuleval \
    --agent cascaded.py \
    --wait-k 2 \
    --source-segment-size 1000 \
    --source en_et/source.txt \
    --source-language en \
    --target en_et/target.txt \
    --target-language et \
    --output output/en_et \
    --device cuda:0 \
    --silence-limit-ms 500

simuleval \
    --agent cascaded.py \
    --wait-k 2 \
    --source-segment-size 1000 \
    --source en_ru/source.txt \
    --source-language en \
    --target en_ru/target.txt \
    --target-language ru \
    --output output/en_ru \
    --device cuda:0 \
    --silence-limit-ms 500

simuleval \
    --agent cascaded.py \
    --wait-k 2 \
    --source-segment-size 1000 \
    --source et_en/source.txt \
    --source-language et \
    --target et_en/target.txt \
    --target-language en \
    --output output/et_en \
    --device cuda:0 \
    --silence-limit-ms 500

simuleval \
    --agent cascaded.py \
    --wait-k 2 \
    --source-segment-size 1000 \
    --source et_ru/source.txt \
    --source-language et \
    --target et_ru/target.txt \
    --target-language ru \
    --output output/et_ru \
    --device cuda:0 \
    --silence-limit-ms 500

simuleval \
    --agent cascaded.py \
    --wait-k 2 \
    --source-segment-size 1000 \
    --source ru_en/source.txt \
    --source-language ru \
    --target ru_en/target.txt \
    --target-language en \
    --output output/ru_en \
    --device cuda:0 \
    --silence-limit-ms 500

simuleval \
    --agent cascaded.py \
    --wait-k 2 \
    --source-segment-size 1000 \
    --source ru_et/source.txt \
    --source-language ru \
    --target ru_et/target.txt \
    --target-language et \
    --output output/ru_et \
    --device cuda:0 \
    --silence-limit-ms 500
