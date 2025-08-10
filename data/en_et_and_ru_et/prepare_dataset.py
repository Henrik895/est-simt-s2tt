import argparse
import json
import logging
import os
import torchaudio

from random import randint


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger("prepare_dataset")


FILES_PATH = "" # Where source audio files can be found
TRANSLATIONS_PATH = "" # Where the translated transcriptions of source audio files can be found

LANG_EST = "est"
LANG_ENG = "eng"
LANG_RUS = "rus"

class SpeechToTextPair():
    def __init__(self, source_name: str, target_name: str, source_lang: str, target_lang: str, source_meta: str, target_meta: str):
        self.name = source_name
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.source_path = f"{FILES_PATH}/{source_name}"
        
        assert source_lang != LANG_EST
        self.target_path = f"{TRANSLATIONS_PATH}/{'en' if source_lang == LANG_ENG else 'ru'}/{target_name}"
        
        self.source_meta = f"{self.target_path}/{source_meta}"
        self.target_meta = f"{self.target_path}/{target_meta}"

# ENG-EST datasets

# 13100 samples
DATA_LJSPEECH = SpeechToTextPair(
    "LJSpeech-1.1",
    "LJSpeech-1.1",
    LANG_ENG,
    LANG_EST,
    "metadata.csv",
    "metadata-et.csv",
)

# train-360 95404 samples
DATA_LIBRISPEECH = SpeechToTextPair(
    "LibriSpeech",
    "LibriSpeech",
    LANG_ENG,
    LANG_EST,
    "train-360.csv",
    "train-360-et.csv",
)

# 114813 samples
DATA_TEDLIUM = SpeechToTextPair(
    "TEDLIUM_release-3",
    "TEDLIUM_release-3",
    LANG_ENG,
    LANG_EST,
    "metadata-clean.csv",
    "metadata-clean-et.csv",
)

# 1845369 samples
DATA_CV_CORPUS = SpeechToTextPair(
    "cv-corpus-21.0-2025-03-14",
    "cv-corpus-21.0-2025-03-14",
    LANG_ENG,
    LANG_EST,
    "metadata.csv",
    "metadata-et.csv",
)

# 300076 samples
DATA_TATOEBA = SpeechToTextPair(
    "tatoeba_audio_eng",
    "tatoeba_audio_eng",
    LANG_ENG,
    LANG_EST,
    "metadata.csv",
    "metadata-et.csv",
)

# 182466 samples
DATA_VOXPOPULI = SpeechToTextPair(
    "voxpopuli",
    "voxpopuli",
    LANG_ENG,
    LANG_EST,
    "train.csv",
    "train-et.csv",
)

# 2551228 samples
DATASETS_ENG_EST = [
    DATA_LJSPEECH, DATA_LIBRISPEECH,
    DATA_TEDLIUM, DATA_CV_CORPUS,
    DATA_TATOEBA, DATA_VOXPOPULI,
]

# RUS-EST datasets

# 168633 samples
DATA_CV_CORPUS_RU = SpeechToTextPair(
    "cv-corpus-20.0-2024-12-06",
    "cv-corpus-20.0-2024-12-06",
    LANG_RUS,
    LANG_EST,
    "metadata.csv",
    "metadata-et.csv",
)

# 1094017 samples
DATA_GOLOS_OPUS = SpeechToTextPair(
    "golos_opus",
    "golos_opus",
    LANG_RUS,
    LANG_EST,
    "train.csv",
    "train-et.csv",
)

# 1149404 samples
DATA_BURIY_AUDIOBOOKS = SpeechToTextPair(
    "private_buriy_audiobooks_2",
    "private_buriy_audiobooks_2",
    LANG_RUS,
    LANG_EST,
    "metadata.csv",
    "metadata-et.csv",
)

# 1410911 samples
DATA_PUBLIC_YOUTUBE = SpeechToTextPair(
    "public_youtube1120",
    "public_youtube1120",
    LANG_RUS,
    LANG_EST,
    "metadata.csv",
    "metadata-et.csv",
)

# 651645 samples
DATA_RADIO_2 = SpeechToTextPair(
    "radio_2",
    "radio_2",
    LANG_RUS,
    LANG_EST,
    "metadata.csv",
    "metadata-et.csv",
)

# 50260 samples
DATA_RULS = SpeechToTextPair(
    "ruls",
    "ruls",
    LANG_RUS,
    LANG_EST,
    "train-clean.csv",
    "train-clean-et.csv",
)

# 29161 samples
DATA_TEDX_RU = SpeechToTextPair(
    "tedx-ru",
    "tedx_ru",
    LANG_RUS,
    LANG_EST,
    "train.csv",
    "train-et.csv",
)

# 4554031 samples
DATASETS_RUS_EST = [
    DATA_CV_CORPUS_RU, DATA_GOLOS_OPUS,
    DATA_BURIY_AUDIOBOOKS, DATA_PUBLIC_YOUTUBE,
    DATA_RADIO_2, DATA_RULS,
    DATA_TEDX_RU,
]

# Combined data

DATASETS = DATASETS_ENG_EST + DATASETS_RUS_EST

parser = argparse.ArgumentParser(
    description=(
        "Prepares dataset for finetuning. Uses suitable format for Seamless finetuning."
    )
)
parser.add_argument(
    "--limit", # can set limits if smaller dataset required
    type=int,
    required=False,
)
parser.add_argument(
    "--cut", # create partial sequences
    required=False,
    default=False,
    action="store_true",
)
parser.add_argument(
    "--manifest", # file created in the end
    type=str,
    required=False,
    default="train_manifest.json",
)
parser.add_argument(
    "--audio", # where to store resampled and partial sequence audio
    type=str,
    required=True,
)

args = parser.parse_args()

limit = args.limit
cut = args.cut
manifest_path = args.manifest
audio = args.audio

logger.info("Run parameters")
logger.info(f"limit: {limit}")
logger.info(f"cut: {cut}")
logger.info(f"manifest: {manifest_path}")
logger.info(f"audio: {audio}")

logger.info("Datasets info")
logger.info(f"ENG-EST datasets: {len(DATASETS_ENG_EST)}")
logger.info(f"RUS-EST datasets: {len(DATASETS_RUS_EST)}")

logger.info("Starting processing datasets")

total_eng_est_samples = 0
total_rus_est_samples = 0

total_eng_est_len = 0
total_rus_est_len = 0

with open(manifest_path, "w", encoding="utf-8") as mf:
    source_counter = 0
    target_counter = 0
    for ds in DATASETS:
        logger.info(f"Processing {ds.source_lang}-{ds.target_lang} dataset {ds.name}")
        resampled_dir = f"{audio}/{ds.name}"
        os.mkdir(resampled_dir) # Directory for storing resampled files
        logger.info(f"Created directory {resampled_dir} for storing resampled audio files")
        cuts_dir = None
        if cut is True:
            cuts_dir = f"{resampled_dir}/cuts"
            os.mkdir(cuts_dir)
            logger.info(f"Created directory {cuts_dir} for storing cut audio files")
        sources = {}
        cuts = []
        skipped = []
        skipped_count = 0
        missed = [] # Sometimes corresponding audio file was missing
        missing_count = 0
        ds_source_counter = 0
        with open(ds.source_meta, "r", encoding="utf-8") as f:
            for line in f:
                clip, text = line.strip().split("|")
                audio_path = f"{ds.source_path}/{clip}"
                try:
                    waveform, sampling_rate = torchaudio.load(audio_path)
                except:
                    logger.error(f"Failed to open file: {audio_path}")
                    missed.append(clip)
                    missing_count += 1
                    continue
                source_audio_path = audio_path
                if sampling_rate != 16000:
                    waveform = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)(waveform)
                    sampling_rate = 16000
                    source_audio_path = f"{resampled_dir}/{clip.replace('/', '_')}"
                    if source_audio_path.endswith(".flac"):
                        source_audio_path = source_audio_path.replace(".flac", ".wav")
                    if source_audio_path.endswith(".ogg"):
                        source_audio_path = source_audio_path.replace(".ogg", ".wav")
                    if source_audio_path.endswith(".mp3"):
                        source_audio_path = source_audio_path.replace(".mp3", ".wav")
                    if source_audio_path.endswith(".opus"):
                        source_audio_path = source_audio_path.replace(".opus", ".wav")
                    if not source_audio_path.endswith(".wav"):
                        raise ValueError(f"Unknown file extension: {source_audio_path}")
                    torchaudio.save(source_audio_path, waveform, sample_rate=sampling_rate)
                waveform = waveform.transpose(0, 1)

                # If the length is less than 1s then skip the file
                if waveform.size()[0] < 16000:
                    logger.warning(f"Skipping file {audio_path}, length {waveform.size()[0] / 16000}s")
                    skipped.append(clip)
                    skipped_count += 1
                    continue

                # If the length is more than 30s then skip the file also
                if waveform.size()[0] > 30*16000:
                    logger.warning(f"Skipping file {audio_path}, length {waveform.size()[0] / 16000}s")
                    skipped.append(clip)
                    skipped_count += 1
                    continue

                if ds.source_lang == LANG_ENG:
                    total_eng_est_len += (waveform.size()[0] / sampling_rate)
                elif ds.source_lang == LANG_RUS:
                    total_rus_est_len += (waveform.size()[0] / sampling_rate)
                else:
                    raise ValueError(f"Unexpected source language {ds.source_lang}") 

                sources[source_counter] = {
                    "id": source_counter,
                    "lang": ds.source_lang,
                    "text": text,
                    "audio_local_path": source_audio_path,
                    "waveform": None,
                    "sampling_rate": 16000,
                    "units": [],
                }
                source_counter += 1

                if cut is True:
                    cut_length = randint(10, 40) / 100
                    cut_waveform = waveform[:int(waveform.size()[0] * cut_length), ...]
                    cut_audio_path = f"{cuts_dir}/{clip.replace('/', '_')}"
                    if cut_audio_path.endswith(".flac"):
                        cut_audio_path = cut_audio_path.replace(".flac", ".wav")
                    if cut_audio_path.endswith(".ogg"):
                        cut_audio_path = cut_audio_path.replace(".ogg", ".wav")
                    if cut_audio_path.endswith(".mp3"):
                        cut_audio_path = cut_audio_path.replace(".mp3", ".wav")
                    if cut_audio_path.endswith(".opus"):
                        cut_audio_path = cut_audio_path.replace(".opus", ".wav")
                    if not cut_audio_path.endswith(".wav"):
                        raise ValueError(f"Unknown file extension: {cut_audio_path}")
                    torchaudio.save(cut_audio_path, cut_waveform, sample_rate=sampling_rate, channels_first=False)
                    cut_text = text[:int(len(text) * cut_length)]
                    cuts.append(cut_length)
                    sources[source_counter] = {
                        "id": source_counter,
                        "lang": ds.source_lang,
                        "text": cut_text,
                        "audio_local_path": cut_audio_path,
                        "waveform": None,
                        "sampling_rate": 16000,
                        "units": [],
                    }
                    source_counter += 1

                ds_source_counter += 1
                if source_counter % 1000 == 0:
                    logger.info(f"Total processed source sentences {source_counter}")
                if ds_source_counter % 500 == 0:
                    logger.info(f"Dataset processed source sentences {ds_source_counter}")
                if limit is not None:
                    if ds_source_counter % limit == 0:
                        break

        targets = {}
        ds_target_counter = 0
        with open(ds.target_meta, "r", encoding="utf-8") as f:
            for line in f:
                clip, text = line.strip().split("|")
                if clip in skipped:
                    continue
                if clip in missed:
                    continue
                # For S2TT finetuning it is not important for target sentence to include
                # audio and extracted units
                targets[target_counter] = {
                    "id": target_counter,
                    "lang": ds.target_lang,
                    "text": text,
                    "audio_local_path": None,
                    "waveform": None,
                    "sampling_rate": None,
                    "units": None,
                }
                target_counter += 1

                if cut is True:
                    cut_length = cuts.pop(0)
                    cut_text = text[:int(len(text) * cut_length)]
                    targets[target_counter] = {
                        "id": target_counter,
                        "lang": ds.target_lang,
                        "text": cut_text,
                        "audio_local_path": None,
                        "waveform": None,
                        "sampling_rate": None,
                        "units": None,
                    }
                    target_counter += 1

                ds_target_counter += 1
                if target_counter % 1000 == 0:
                    logger.info(f"Processed target sentences {target_counter}")
                if ds_target_counter % 500 == 0:
                    logger.info(f"Dataset processed source sentences {ds_target_counter}")
                if limit is not None:
                    if ds_target_counter % limit == 0:
                        break

        assert source_counter == target_counter # These should always be equal

        for key in sources.keys():
            sample = {
                "source": sources[key],
                "target": targets[key],
            }
            mf.write(json.dumps(sample) + "\n")

        if ds.source_lang == LANG_ENG:
            total_eng_est_samples += len(sources.keys())
        elif ds.source_lang == LANG_RUS:
            total_rus_est_samples += len(sources.keys())
        else:
            raise ValueError(f"Unexpected source language {ds.source_lang}")

        logger.info(f"Completed processing {ds.source_lang}-{ds.target_lang} dataset {ds.name}")
        logger.info(f"Samples added {len(sources.keys())}")
        logger.info(f"Samples skipped {skipped_count}")
        logger.info(f"Samples missing: {missing_count}")

logger.info("Dataset processing completed")
logger.info(f"Total samples processed: {total_eng_est_samples + total_rus_est_samples}")
logger.info(f"ENG-EST samples processed: {total_eng_est_samples}")
logger.info(f"ENG-EST samples length: {total_eng_est_len / 60} minutes")
logger.info(f"ENG-EST samples length: {total_eng_est_len / (60 * 60)} hours")
logger.info(f"RUS-EST processed: {total_rus_est_samples}")
logger.info(f"RUS-EST samples length: {total_rus_est_len / 60} hours")
logger.info(f"RUS-EST samples length: {total_rus_est_len / (60 * 60)} hours")
