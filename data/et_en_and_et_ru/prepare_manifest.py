import argparse
import json
import logging

LANG_EST = "est"
LANG_ENG = "eng"
LANG_RUS = "rus"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger("prepare_est_speech_finetuning_dataset")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", # Input from cut_audio.py
    type=str,
    required=True,
)
parser.add_argument(
    "--output", # The target file name
    type=str,
    default="manifest.json",
    required=False,
)
parser.add_argument(
    "--audio-base", # Base folder where audio files are (the input has relative paths)
    type=str,
    default="",
    required=False,
)
parser.add_argument(
    "--offset", # If you want to create multiple files, then to avoid overlapping ids
    type=int,
    default=0,
    required=False,
)
args = parser.parse_args()

input = args.input
output = args.output
audio_base = args.audio_base
offset = args.offset

logger.info("Creating est-eng/rus manifest")
logger.info(f"Input: {input}")
logger.info(f"Output: {output}")
logger.info(f"Audio base: {audio_base}")
logger.info(f"Offset: {offset}")

with open(output, "w", encoding="utf-8") as mf:
    with open(input, "r", encoding="utf-8") as f:
        counter = offset
        for line in f:
            item = line.strip().split("|")

            audio_path = f"{audio_base}/{item[-1]}"

            # Format, which can be used for Seamless fine-tuning
            source = {
                "id": counter,
                "lang": LANG_EST,
                "text": item[3],
                "audio_local_path": audio_path,
                "waveform": None,
                "sampling_rate": 16000,
                "units": None,
            }

            target = {
                "id": counter,
                "lang": LANG_ENG,
                "text": item[4],
                "audio_local_path": None,
                "waveform": None,
                "sampling_rate": None,
                "units": None,
            }

            sample = {
                "source": source,
                "target": target,
            }

            mf.write(json.dumps(sample) + "\n")

            counter += 1

            source["id"] = counter
            target["id"] = counter
            target["lang"] = LANG_RUS
            target["text"] = item[5]

            sample = {
                "source": source,
                "target": target,
            }

            mf.write(json.dumps(sample) + "\n")

            counter += 1

logger.info("Finished creating manifest")
logger.info(f"Added {counter} samples to manifest")
