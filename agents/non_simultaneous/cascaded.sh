simuleval \
    --agent cascaded.py \
    --source-segment-size 1000 \
    --source en_et/source.txt \
    --source-language en \
    --target en_et/target.txt \
    --target-language et \
    --output output_cascaded/en_et \
    --device cuda:0

simuleval \
    --agent cascaded.py \
    --source-segment-size 1000 \
    --source en_ru/source.txt \
    --source-language en \
    --target en_ru/target.txt \
    --target-language ru \
    --output output_cascaded/en_ru \
    --device cuda:0

simuleval \
    --agent cascaded.py \
    --source-segment-size 1000 \
    --source et_en/source.txt \
    --source-language et \
    --target et_en/target.txt \
    --target-language en \
    --output output_cascaded/et_en \
    --device cuda:0

simuleval \
    --agent cascaded.py \
    --source-segment-size 1000 \
    --source et_ru/source.txt \
    --source-language et \
    --target et_ru/target.txt \
    --target-language ru \
    --output output_cascaded/et_ru \
    --device cuda:0

simuleval \
    --agent cascaded.py \
    --source-segment-size 1000 \
    --source ru_en/source.txt \
    --source-language ru \
    --target ru_en/target.txt \
    --target-language en \
    --output output_cascaded/ru_en \
    --device cuda:0

simuleval \
    --agent cascaded.py \
    --source-segment-size 1000 \
    --source ru_et/source.txt \
    --source-language ru \
    --target ru_et/target.txt \
    --target-language et \
    --output output_cascaded/ru_et \
    --device cuda:0
