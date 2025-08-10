"""
Prepares the data for SimulEval evaluation

Creates 6 language pairs:
  - est-eng
  - est-rus
  - eng-est
  - eng-rus
  - rus-est
  - rus-eng

Each pair will have following files associated with it:
  - source.txt - text file with a source file path on each row
  - target.txt - text file with reference text (translation) on each row
  - tgt_lang.txt - text file with target language on each row
"""
from pathlib import Path
import json
import shutil

FOLDER_DELETE_ENABLED = False

LANGUAGE_EST = {
    "code": "et",
    "folder": "data_est",
}

LANGUAGE_ENG = {
    "code": "en",
    "folder": "data_eng",
}

LANGUAGE_RUS = {
    "code": "ru",
    "folder": "data_rus",
}

LANGUAGES = [
    LANGUAGE_EST,
    LANGUAGE_ENG,
    LANGUAGE_RUS,
]

completed = []

for lang_a in LANGUAGES:
    for lang_b in LANGUAGES:
        if lang_a["code"] == lang_b["code"]:
            continue

        pair = f"{lang_a['code']}_{lang_b['code']}"
        pair_opposite = f"{lang_b['code']}_{lang_a['code']}"

        if pair in completed or pair_opposite in completed:
            continue

        # Prepare folders
        try:
            Path.mkdir(Path(f"{lang_a['code']}_{lang_b['code']}"), parents=True, exist_ok=False)
        except FileExistsError as e:
            print(f"{lang_a['code']}_{lang_b['code']} already exists")
            if FOLDER_DELETE_ENABLED:
                print("Deleting existing files and creating empty directory")
                shutil.rmtree(Path(f"{lang_a['code']}_{lang_b['code']}"))
                Path.mkdir(Path(f"{lang_a['code']}_{lang_b['code']}"), parents=True, exist_ok=False)
            else:
                print("Automatic deletion disabled")
                raise e
            
        try:
            Path.mkdir(Path(f"{lang_b['code']}_{lang_a['code']}"), parents=True, exist_ok=False)
        except FileExistsError as e:
            print(f"{lang_b['code']}_{lang_a['code']} already exists")
            if FOLDER_DELETE_ENABLED:
                print("Deleting existing files and creating empty directory")
                shutil.rmtree(Path(f"{lang_b['code']}_{lang_a['code']}"))
                Path.mkdir(Path(f"{lang_b['code']}_{lang_a['code']}"), parents=True, exist_ok=False)
            else:
                print("Automatic deletion disabled")
                raise e

        # Create data
    
        with open(f"{lang_a['folder']}/data.json", "r", encoding="utf-8") as f:
            data_a = {data["id"]:{k:v for k,v in data.items() if k != "id"} for data in json.load(f)}
        
        with open(f"{lang_b['folder']}/data.json", "r", encoding="utf-8") as f:
            data_b = {data["id"]:{k:v for k,v in data.items() if k != "id"} for data in json.load(f)}

        source_a = []
        source_b = []

        target_a = []
        target_b = []

        tgt_a = []
        tgt_b = []

        for k in data_a.keys():
            if data_b.get(k, None) is None:
                continue

            sample_a = data_a[k]
            sample_b = data_b[k]

            source_a.append(f"{lang_a['folder']}/{sample_a['path']}")
            source_b.append(f"{lang_b['folder']}/{sample_b['path']}")

            target_a.append(sample_a["raw_transcription"])
            target_b.append(sample_b["raw_transcription"])

            tgt_a.append(lang_a["code"])
            tgt_b.append(lang_b["code"])

        # Write lang_a - lang_b pair data

        with open(f"{lang_a['code']}_{lang_b['code']}/source.txt", "w", encoding="utf-8") as f:
            for source in source_a:
                f.write(f"{source}\n")

        with open(f"{lang_a['code']}_{lang_b['code']}/target.txt", "w", encoding="utf-8") as f:
            for target in target_b:
                f.write(f"{target}\n")

        with open(f"{lang_a['code']}_{lang_b['code']}/tgt_lang.txt", "w", encoding="utf-8") as f:
            for tgt in tgt_b:
                f.write(f"{tgt}\n")

        # Write lang_b - lang_a pair data
        with open(f"{lang_b['code']}_{lang_a['code']}/source.txt", "w", encoding="utf-8") as f:
            for source in source_b:
                f.write(f"{source}\n")

        with open(f"{lang_b['code']}_{lang_a['code']}/target.txt", "w", encoding="utf-8") as f:
            for target in target_a:
                f.write(f"{target}\n")

        with open(f"{lang_b['code']}_{lang_a['code']}/tgt_lang.txt", "w", encoding="utf-8") as f:
            for tgt in tgt_a:
                f.write(f"{tgt}\n")

        completed.append(pair)
        completed.append(pair_opposite)
