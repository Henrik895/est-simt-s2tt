import argparse
import logging
import soundfile

from random import randint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger("cut_audio_est_speech_dataset")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base", # Location of the original audio files
    type=str,
    required=True,
)
parser.add_argument(
    "--input", # Output from compile.py/filter.py/sample.py
    type=str,
    required=True,
)
parser.add_argument(
    "--output", # Where to store the cut audio files
    type=str,
    required=True,
)
parser.add_argument(
    "--file",
    type=str,
    default="train_audio.txt", # Where to store the text file with references to cut audio
    required=False,
)
parser.add_argument(
    "--partial", # Create partial sequences from full sequences
    action="store_true",
    default=False,
)
args = parser.parse_args()

base = args.base
input = args.input
output = args.output
file = args.file
partial = args.partial

logger.info("Creating audio files based on samples")
logger.info(f"Base path: {base}")
logger.info(f"Input file: {input}")
logger.info(f"Output folder: {output}")

with open(file, "w", encoding="utf-8") as ff:
    with open(input, "r", encoding="utf-8") as f:
        counter = 0
        for line in f:
            item = line.strip().split("|")

            filename = item[0]
            filepath = f"{base}/{filename}"

            audio, sample_rate = soundfile.read(filepath)
            # This is for error handling because all files should already use 16000hz sampling rate
            if sample_rate != 16000:
                logger.error(f"Unsupported sample rate {sample_rate}")
                logger.error(f"File {filepath}")
                raise ValueError(f"Unsupported sample rate")

            start = int(float(item[1]) * sample_rate)
            end = start + int(float(item[2]) * sample_rate) + 1
            chunk = audio[start:end]
            chunk_path = f"{output}/chunk_{counter}.wav"

            soundfile.write(chunk_path, chunk, sample_rate)
            meta = f"{line.strip()}|{chunk_path}\n"
            ff.write(meta)

            counter += 1

            if partial is True:
                partial_cut = randint(10, 40) / 100
                partial_chunk = chunk[:int(len(chunk)*partial_cut)]
                partial_path = f"{output}/chunk_{counter}.wav"
                meta = meta.strip().split("|")[:-1] # Remove old chunk path
                meta.append(partial_path)
                meta[1] = str(float(meta[1]) * partial_cut)
                meta[2] = str(float(meta[2]) * partial_cut)
                meta[3] = meta[3][:int(len(meta[3])*partial_cut)] #est text
                meta[4] = meta[4][:int(len(meta[4])*partial_cut)] #eng text
                meta[5] = meta[5][:int(len(meta[5])*partial_cut)] #rus text

                meta = f"{'|'.join(meta)}\n"

                soundfile.write(partial_path, partial_chunk, sample_rate)
                ff.write(meta)

                counter += 1

            if counter % 100 == 0:
                logger.info(f"Audio files processed {counter}")
