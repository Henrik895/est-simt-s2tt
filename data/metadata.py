import argparse
import json
import logging
import torchaudio


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger("metadata")

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    type=str,
    required=True,
)

args = parser.parse_args()

input_file = args.input

logger.info(f"Calculating {input_file} metadata")

eng_datasets = [
    "LJSpeech-1.1", "LibriSpeech", "TEDLIUM_release-3",
    "cv-corpus-21.0-2025-03-14", "tatoeba_audio_eng",
    "voxpopuli",
]
rus_datasets = [
    "cv-corpus-20.0-2024-12-06", "golos_opus",
    "private_buriy_audiobooks_2", "public_youtube1120",
    "radio_2", "ruls", "tedx-ru",
]

est_eng = []
est_rus = []

eng_est = {}
for ds in eng_datasets:
    eng_est[ds] = []

rus_est = {}
for ds in rus_datasets:
    rus_est[ds] = []


with open(input_file, "r", encoding="utf-8") as f:
    counter = 0
    for line in f:
        item = json.loads(line.strip())
        source = item["source"]
        target = item["target"]

        source_lang = source["lang"]
        target_lang = target["lang"]

        source_audio = source["audio_local_path"]

        audio, sampling_rate = torchaudio.load(source_audio)
        audio_length = audio.size()[1] / sampling_rate

        if source_lang == "eng":
            found = False
            for ds in eng_est.keys():
                if ds in source_audio:
                    eng_est[ds].append(audio_length)
                    found = True
                    break
            if found != True:
                raise ValueError(f"Unknown source audio: {source_audio}")
        elif source_lang == "rus":
            found = False
            for ds in rus_est.keys():
                if ds in source_audio:
                    rus_est[ds].append(audio_length)
                    found = True
                    break
            if found != True:
                raise ValueError(f"Unknown dataset: {source_audio}")
        elif target_lang == "eng":
            est_eng.append(audio_length)
        elif target_lang == "rus":
            est_rus.append(audio_length)
        else:
            raise ValueError(f"Unexpected language pair: {source_lang}-{target_lang}")

        counter += 1

        if counter % 25000 == 0:
            logger.info(f"{counter} samples processed")


try:
    logging.info(f"Estonian-English samples: {len(est_eng)}")
    logging.info(f"Estonian-English length (s): {sum(est_eng)}")
    logging.info(f"Estonian-English avg. (s): {sum(est_eng) / len(est_eng)}\n")
except:
    logging.info("Estonian-English metadata not available")

try:
    logging.info(f"Estonian-Russian samples: {len(est_rus)}")
    logging.info(f"Estonian-Russian length (s): {sum(est_rus)}")
    logging.info(f"Estonian-Russian avg. (s): {sum(est_rus) / len(est_rus)}\n")
except:
    logging.info("Estonian-Russian metadata not available")


try:
    eng_samples = 0
    eng_length = 0
    logging.info("English-Estonian dataset samples")
    for ds in eng_est.keys(): # Can also get dataset specific information in this case
        try:
            logging.info(f"Samples {ds}: {len(eng_est[ds])}")
            eng_samples += len(eng_est[ds])
            logging.info(f"Length {ds}: {sum(eng_est[ds])}")
            eng_length += sum(eng_est[ds])
            logging.info(f"Avg. (s) {ds}: {sum(eng_est[ds]) / len(eng_est[ds])}")
        except:
            logging.info("Can't be calculated")

    logging.info(f"Total samples: {eng_samples}")
    logging.info(f"Length: {eng_length}")
    logging.info(f"Avg. (s): {eng_length / eng_samples}\n")
except:
    logging.info("English-Estonian metadata not available")


try:
    rus_samples = 0
    rus_length = 0
    logging.info("Russian-Estonian dataset samples")
    for ds in rus_est.keys():
        try:
            logging.info(f"{ds}: {len(rus_est[ds])}")
            rus_samples += len(rus_est[ds])
            logging.info(f"Length {ds}: {sum(rus_est[ds])}")
            rus_length += sum(rus_est[ds])
            logging.info(f"Avg. (s) {ds}: {sum(rus_est[ds]) / len(rus_est[ds])}")
        except:
            logging.info("Can't be calculated")

    logging.info(f"Total samples: {rus_samples}")
    logging.info(f"Length: {rus_length}")
    logging.info(f"Avg. (s): {rus_length / rus_samples}\n")

    logging.info("Calculating combined statistics")
except:
    logging.info("Russian-Estonian metadata not available")


logging.info("Calculating total metadata")
try:
    total_samples = len(est_eng) + len(est_rus) + eng_samples + rus_samples
    total_length = sum(est_eng) + sum(est_rus) + eng_length + rus_length
    avg_length = total_length / total_samples

    logging.info(f"Samples: {total_samples}")
    logging.info(f"Length: {total_length}")
    logging.info(f"Avg. (s): {avg_length}")
except:
    logging.info("Total metadata not available")
