# --cut creates partial sequences and --audio is the target folder
# for partial and resampled audio files
python prepare_dataset.py \
	--manifest partial_manifest.json \
	--audio audio \
	--cut