import argparse
import logging
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger("sample_est_speech_dataset")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", # Output from compile.py or filter.py
    type=str,
    required=True,
)
parser.add_argument(
    "--output", # Where to store the samples
    type=str,
    default="sample.txt",
    required=False,
)
parser.add_argument(
    "--samples", # Number of samples to take
    type=int,
    default=100,
    required=False
)
args = parser.parse_args()

input = args.input
output = args.output
samples = args.samples

logger.info(f"Sampling {samples} samples from {input} to {output}")

with open(input, "r", encoding="utf-8") as f:
    lines = sum(1 for _ in f)

logger.info(f"Detected {lines} samples in file {input}")

if lines < samples:
    logger.info(f"Restricting the samples {samples} to {lines}")
    samples = lines

logger.info(f"Sampling {samples} from {lines} lines")

selected_lines = random.sample(range(0, lines + 1), samples)

logger.info(f"Sampling order generated")

with open(output, "w", encoding="utf-8") as of:
    with open(input, "r", encoding="utf-8") as f:
        count = 0
        for line in f:
            if (samples == lines) or (count in selected_lines):
                of.write(line)
            count += 1

            if count % 1000 == 0:
                logger.info(f"Processing count {count}")

logger.info("Finished sampling")
