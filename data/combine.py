import json
import logging

from collections import defaultdict
from random import shuffle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger("combine_data")

first_file = "" # Contained Estonian-English and Estonian-Russian samples
second_file = "" # Contained English-Estonian and Russian-Estonian samples

est_eng = []
est_rus = []

logger.info(f"Processing {first_file}")
with open(first_file, "r", encoding="utf-8") as f:
    counter = 0
    for line in f:
        item = json.loads(line.strip())
        target_lang = item["target"]["lang"]
        if target_lang == "eng":
            est_eng.append(line)
        elif target_lang == "rus":
            est_rus.append(line)
        else:
            raise ValueError(f"Unknown target language: {target_lang}")
        
        counter += 1

        if counter % 100000 == 0:
            logger.info(f"Processed rows: {counter}")

logger.info(f"Completed processing {first_file}")
logger.info(f"est_eng samples: {len(est_eng)}")
logger.info(f"est_rus samples: {len(est_rus)}")
shuffle(est_eng) # Random shuffle without setting seed
shuffle(est_rus)

assert len(est_eng) == len(est_rus)

num_required_sentences = len(est_eng) # How many English-Estonian and Russian-Estonian samples needed

# We want to have a 1:1 mix of full and partial sequences,
# so we keep separate track of them in the dictionaries.
eng_pairs = defaultdict(lambda: ({"full": [], "cut": []}))
rus_pairs = defaultdict(lambda: ({"full": [], "cut": []}))

eng_datasets = [
    "LJSpeech-1.1", "LibriSpeech", "TEDLIUM_release-3",
    "cv-corpus-21.0-2025-03-14", "tatoeba_audio_eng",
    "voxpopuli",
]
rus_datasets = [
    "cv-corpus-20.0-2024-12-06", "golos_opus",
    "private_buriy_audiobooks_2", "public_youtube1120",
    "radio_2", "ruls", "tedx-ru"
]

# How many samples required per data source
# Divided by two because full and cut separate
num_ds_samples_eng = int(len(est_eng) / len(eng_datasets) / 2)
num_ds_samples_rus = int(len(est_rus) / len(rus_datasets) / 2)

logger.info(f"Needed eng samples for each ds split: {num_ds_samples_eng}")
logger.info(f"Needed rus samples for each ds split: {num_ds_samples_rus}")

def is_english_dataset(path):
    for ds in eng_datasets:
        if ds in path:
            return True, ds
    return False, None

def is_russian_dataset(path):
    for ds in rus_datasets:
        if ds in path:
            return True, ds
    return False, None

logger.info(f"Processing {second_file}")
with open(second_file, "r", encoding="utf-8") as f:
    counter = 0
    for line in f:
        item = json.loads(line.strip())
        audio_path = item["source"]["audio_local_path"]

        if (english_ds := is_english_dataset(audio_path))[0] == True:
            if "/cuts/" in audio_path: # Partial sequence
                eng_pairs[english_ds[1]]["cut"].append(line)
            else:
                eng_pairs[english_ds[1]]["full"].append(line)
        elif (russian_ds := is_russian_dataset(audio_path))[0] == True:
            if "/cuts/" in audio_path:
                rus_pairs[russian_ds[1]]["cut"].append(line)
            else:
                rus_pairs[russian_ds[1]]["full"].append(line)
        else:
            raise ValueError(f"Unknown dataset: {audio_path}")

        counter += 1

        if counter % 100000 == 0:
            logger.info(f"Processed rows: {counter}")

logger.info(f"Completed processing {second_file}")
logger.info("Creating english-estonian pairs")

eng_samples = []
remaining_samples = []

for k in eng_pairs.keys():
    full = eng_pairs[k]["full"]
    cut = eng_pairs[k]["cut"]
    shuffle(full)
    shuffle(cut)

    limit = num_ds_samples_eng
    if len(full) < limit:
        logger.info(f"not enough samples: {k}")
        eng_samples.extend(full)
        eng_samples.extend(cut)
    else:
        logger.info(f"enough samples: {k}")
        eng_samples.extend(full[:limit])
        eng_samples.extend(cut[:limit])
        remaining_samples.extend(full[limit:])
        remaining_samples.extend(cut[limit:])

logger.info(f"Taken samples: {len(eng_samples)}")
logger.info(f"Eng remaining: {len(remaining_samples)}")
num_missing_samples = num_required_sentences - len(eng_samples)
logger.info(f"Num missing samples: {num_missing_samples}")
shuffle(remaining_samples)
eng_samples.extend(remaining_samples[:num_missing_samples])
shuffle(eng_samples)

logger.info(f"Total eng-est samples taken: {len(eng_samples)}")

logger.info("Creating russian-estonian pairs")

remaining_samples = []
rus_samples = []

for k in rus_pairs.keys():
    full = rus_pairs[k]["full"]
    cut = rus_pairs[k]["cut"]
    shuffle(full)
    shuffle(cut)

    limit = num_ds_samples_rus
    if len(full) < limit:
        logger.info(f"not enough samples: {k}")
        rus_samples.extend(full)
        rus_samples.extend(cut)
    else:
        logger.info(f"enough samples: {k}")
        rus_samples.extend(full[:limit])
        rus_samples.extend(cut[:limit])
        remaining_samples.extend(full[limit:])
        remaining_samples.extend(cut[limit:])

logger.info(f"Taken samples: {len(rus_samples)}")
logger.info(f"Rus remaining: {len(remaining_samples)}")
num_missing_samples = num_required_sentences - len(rus_samples)
logger.info(f"Num missing samples: {num_missing_samples}")
shuffle(remaining_samples)
rus_samples.extend(remaining_samples[:num_missing_samples])
shuffle(rus_samples)

logger.info(f"Total rus-est samples taken: {len(rus_samples)}")

all_samples = est_eng + est_rus + eng_samples + rus_samples
shuffle(all_samples)

logger.info(f"Total dataset length: {len(all_samples)}")

logger.info("Splitting dataset 95/10 to train and eval")
split_index = int(len(all_samples) * 0.95)
logger.info(f"Train will have {split_index} samples")
logger.info(f"Eval will have {len(all_samples) - split_index} samples")

train_samples = all_samples[:split_index]
eval_samples = all_samples[split_index:]

with open("train_manifest.json", "w", encoding="utf-8") as f:
    for sample in train_samples:
        f.write(sample)

with open("eval_manifest.json", "w", encoding="utf-8") as f:
    for sample in eval_samples:
        f.write(sample)
