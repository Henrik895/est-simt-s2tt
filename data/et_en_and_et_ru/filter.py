import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger("filter_est_speech_dataset")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", # File created by compile.py
    type=str,
    required=True,
)
parser.add_argument(
    "--output", # Where to store the output
    type=str,
    default="filtered.txt",
    required=False,
)
parser.add_argument(
    "--min", # Min allowed audio duration
    type=float,
    default=1.0,
    required=False
)
parser.add_argument(
    "--max", # Max allowed audio duration
    type=float,
    default=30.0,
    required=False,
)
args = parser.parse_args()

input = args.input
output = args.output
min_length = args.min
max_length = args.max

logger.info(f"Filtering estonian speech dataset")
logger.info(f"Input file: {input}")
logger.info(f"Output file: {output}")
logger.info(f"Min length: {min_length}s")
logger.info(f"Max length: {max_length}s")

too_short = 0
too_long = 0
total = 0
with open(output, "w", encoding="utf-8") as ff:
    with open(input, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            items = line.strip().split("|")
            time = float(items[2])
            if time < min_length:
                too_short += 1
                continue
            if time > max_length:
                too_long += 1
                continue
            ff.write(line)

logger.info(f"Lines processed: {total}")
logger.info(f"Included: {total - too_long - too_short}")
logger.info(f"Excluded: {too_long +  too_short}")
logger.info(f"Too long: {too_long}")
logger.info(f"Too short: {too_short}")
