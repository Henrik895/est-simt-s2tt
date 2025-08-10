from datasets import load_dataset
from pathlib import Path
import json
import shutil

DATASET = "google/fleurs"
DATASET_SPLIT = "test"

DATA_EST = "et_ee"
DATA_ENG = "en_us"
DATA_RUS = "ru_ru"

FOLDER_EST = "data_est"
FOLDER_ENG = "data_eng"
FOLDER_RUS = "data_rus"

FOLDER_DELETE_ENABLED = True

def create_folder(path, enable_delete = False):
    try:
        Path.mkdir(Path(path) / "audio", parents=True, exist_ok=False)
    except FileExistsError as e:
        print(f"{path}/audio already exists")
        if enable_delete:
            print("Deleting existing files and creating empty directories")
            shutil.rmtree(Path(path))
            Path.mkdir(Path(path) / "audio", parents=True, exist_ok=False)
        else:
            print("Automatic deletion disabled")
            raise e


def create_data_dict(language, split):
    data = load_dataset(DATASET, language, split=split, trust_remote_code=True)
    samples = {}
    for sample in data:
        if samples.get(sample["id"], None) is not None:
            continue

        samples[sample["id"]] = {
            "path": "/".join(sample["path"].split("/")[:-1]) + "/" + sample["audio"]["path"],
            "transcription": sample["transcription"],
            "raw_transcription": sample["raw_transcription"],
        }

    return samples


create_folder(FOLDER_EST, FOLDER_DELETE_ENABLED)
create_folder(FOLDER_ENG, FOLDER_DELETE_ENABLED)
create_folder(FOLDER_RUS, FOLDER_DELETE_ENABLED)

samples_est = create_data_dict(DATA_EST, DATASET_SPLIT)
samples_eng = create_data_dict(DATA_ENG, DATASET_SPLIT)
samples_rus = create_data_dict(DATA_RUS, DATASET_SPLIT)

json_est = []
json_eng = []
json_rus = []

for i, k in enumerate(list(sorted(samples_est.keys()))):
    if samples_eng.get(k, None) is None or samples_rus.get(k, None) is None:
        continue

    file_name = f"sample_{i}.wav"

    # Write Estonian file
    path_est = f"{FOLDER_EST}/audio/{file_name}"
    shutil.copy(samples_est[k]["path"], path_est)

    json_est.append({
        "id": i,
        "path": f"audio/{file_name}",
        "transcription": samples_est[k]["transcription"],
        "raw_transcription": samples_est[k]["raw_transcription"]
    })

    # Write English file
    path_eng = f"{FOLDER_ENG}/audio/{file_name}"
    shutil.copy(samples_eng[k]["path"], path_eng)

    json_eng.append({
        "id": i,
        "path": f"audio/{file_name}",
        "transcription": samples_eng[k]["transcription"],
        "raw_transcription": samples_eng[k]["raw_transcription"]
    })

    # Write Russian file
    path_rus = f"{FOLDER_RUS}/audio/{file_name}"
    shutil.copy(samples_rus[k]["path"], path_rus)

    json_rus.append({
        "id": i,
        "path": f"audio/{file_name}",
        "transcription": samples_rus[k]["transcription"],
        "raw_transcription": samples_rus[k]["raw_transcription"]
    })

with open(f"{FOLDER_EST}/data.json", "w", encoding="utf-8") as f:
    json.dump(json_est, f, indent=2)

with open(f"{FOLDER_ENG}/data.json", "w", encoding="utf-8") as f:
    json.dump(json_eng, f, indent=2)

with open(f"{FOLDER_RUS}/data.json", "w", encoding="utf-8") as f:
    json.dump(json_rus, f, indent=2)
