import argparse
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger("compile_est_speech_dataset")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file", # Where to write the output
    type=str,
    default="compiled.txt",
    required=False,
)
args = parser.parse_args()

filename = args.file

def read_lines(file: str):
    lines = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            lines.append(line)
    return lines


texts_meta = read_lines("translations/train.yaml")
texts_est = read_lines("translations/train.et")
texts_eng = read_lines("translations/train.en")
texts_ru = read_lines("translations/train.ru")

logger.info("Starting to compile Estonian speech dataset")
total = 0
with open(filename, "w", encoding="utf-8") as f:
    for m, et, en, ru in zip(texts_meta, texts_est, texts_eng, texts_ru):
        # Making valid JSON out of the text line
        m = m.replace('duration: ', '"duration": ').replace('offset: ', '"offset": ').replace('wav: ', '"wav": ')[2:]
        meta = json.loads(m)
        wav = meta["wav"]
        off = meta["offset"]
        dur = meta["duration"]
        line = f"{wav}|{off}|{dur}|{et}|{en}|{ru}\n"
        f.write(line)
        total += 1

logger.info(f"Items processed: {total}")
logger.info(f"Results saved to {filename}")

