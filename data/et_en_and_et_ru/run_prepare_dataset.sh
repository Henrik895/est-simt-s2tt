python compile.py \
	--file train_compiled_partial.txt

python filter.py \
	--input train_compiled_partial.txt \
	--output train_filtered_partial.txt

# --samples 10000000 in this case means take everything because
# there are not 10 million samples
python sample.py \
	--input train_filtered_partial.txt \
	--output train_sample_partial.txt \
	--samples 10000000

python cut_audio.py \
	--base /gpfs/helios/home/henrklp/train/audio \
	--input train_sample_partial.txt \
	--output audio_partial \
	--file train_audio_partial.txt \
	--partial

python prepare_manifest.py \
	--input train_audio_partial.txt \
	--output train_manifest_partial.json \
	--audio-base /gpfs/helios/home/henrklp/est-speech-data
