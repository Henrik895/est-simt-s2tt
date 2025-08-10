from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, get_scheduler
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from collections import defaultdict

import argparse
import json
import logging
import sacrebleu
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger("finetune_nllb")

LANG_EST = "est_Latn"
LANG_ENG = "eng_Latn"
LANG_RUS = "rus_Cyrl"

LANG_MAP = {
    "est": LANG_EST,
    "eng": LANG_ENG,
    "rus": LANG_RUS,
}

CHECKPOINT = "facebook/nllb-200-distilled-1.3B"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

parser = argparse.ArgumentParser()

parser.add_argument(
    "--train-manifest",
    type=str,
    required=True,
)

parser.add_argument(
    "--eval-manifest",
    type=str,
    required=True,
)

parser.add_argument(
    "--epochs",
    type=int,
    required=False,
    default=10,
)

parser.add_argument(
    "--batch-size",
    type=int,
    required=False,
    default=4,
)

parser.add_argument(
    "--learning-rate",
    type=int,
    required=False,
    default=1e-6,
)

parser.add_argument(
    "--log-steps",
    type=int,
    required=False,
    default=10,
)

parser.add_argument(
    "--eval-steps",
    type=int,
    required=False,
    default=100,
)

parser.add_argument(
    "--warmup",
    type=int,
    required=False,
    default=0,
)

parser.add_argument(
    "--patience",
    type=int,
    required=False,
    default=5,
)

parser.add_argument(
    "--output",
    type=str,
    required=False,
    default="",
)

parser.add_argument(
    "--store-metrics",
    action="store_true",
    default=False,
)

args = parser.parse_args()

train_manifest = args.train_manifest
eval_manifest = args.eval_manifest
epochs = args.epochs
log_steps = args.log_steps
eval_steps = args.eval_steps
learning_rate = args.learning_rate
batch_size = args.batch_size
warmup = args.warmup
patience = args.patience
output = args.output
store_metrics = args.store_metrics

logger.info("Starting NLLB finetuning")
logger.info(f"starting checkpoint: {CHECKPOINT}")
logger.info(f"using device: {device}")
logger.info("====params====")
logger.info(f"train manifest: {train_manifest}")
logger.info(f"eval manifest: {eval_manifest}")
logger.info(f"epochs: {epochs}")
logger.info(f"log steps: {log_steps}")
logger.info(f"eval steps: {eval_steps}")
logger.info(f"learning rate: {learning_rate}")
logger.info(f"batch size: {batch_size}")
logger.info("scheduler: linear")
logger.info(f"warmup steps: {warmup}")
logger.info("optimizer: AdamW")
logger.info(f"patience: {patience}")
logger.info(f"output destination: {output}")
logger.info(f"store metrics: {store_metrics}")
logger.info("==============")

tokenizer = NllbTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)


def load_translations(filename):
    translations = defaultdict(lambda: [])
    with open(filename, "r", encoding="utf-8") as f:
        logger.info(f"Loading translations from {filename}")
        for line in f:
            item = json.loads(line.strip())
            source_text = item["source"]["text"]
            source_lang = LANG_MAP[item["source"]["lang"]]
            target_text = item["target"]["text"]
            target_lang = LANG_MAP[item["target"]["lang"]]
            translation = { source_lang: source_text, target_lang: target_text }
            translations[f"{source_lang}-{target_lang}"].append(translation)

    total = 0
    for k, v in translations.items():
        logger.info(f"Pair {k} loaded {len(v)} translations")
        total += len(v)

    logger.info(f"From {filename} loaded total {total} translations")

    return translations

train_translations = load_translations(train_manifest)
eval_translations = load_translations(eval_manifest)

def preprocess_dataset(dataset, tokenizer, source_lang, target_lang):
    tokenizer.src_lang = source_lang
    tokenizer.tgt_lang = target_lang

    sources = [sample[source_lang] for sample in dataset["translations"]]
    targets = [sample[target_lang] for sample in dataset["translations"]]

    inputs = tokenizer(sources, text_target=targets, max_length=200, padding=True, truncation=True)

    return inputs


def prepare_tokenized_dataset(dataset):
    dataset = dataset.remove_columns(["translations"])
    dataset.set_format('torch')

    return dataset


def create_dataset(translations):
    sub_datasets = []
    for k, v in translations.items():
        logger.info(f"Processing {k}")
        sub_dataset = Dataset.from_dict({ "translations": v })
        source_lang, target_lang = k.split("-")
        sub_dataset = sub_dataset.map(
            preprocess_dataset, 
            batched=True, 
            fn_kwargs={
                "tokenizer": tokenizer,
                "source_lang": source_lang,
                "target_lang": target_lang,
            },
        )
        sub_dataset = prepare_tokenized_dataset(sub_dataset)
        sub_datasets.append(sub_dataset)

    dataset = concatenate_datasets(sub_datasets)

    return dataset

logger.info("Creating train dataset")
train_dataset = create_dataset(train_translations)
logger.info("Train dataset created")
logger.info(f"Train dataset rows: {train_dataset.num_rows}")

logger.info("Creating eval dataset")
eval_dataset = create_dataset(eval_translations)
logger.info("Eval dataset created")
logger.info(f"Eval dataset rows: {eval_dataset.num_rows}")

logger.info("Creating dataloaders")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=CHECKPOINT, padding=True)

train_dataloader = DataLoader(
    train_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator,
)
logger.info("Train dataloader created")

eval_dataloader = DataLoader(
    eval_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator,
)
logger.info("Eval dataloader created")

model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)
training_steps = epochs * len(train_dataloader)
logger.info(f"Training steps per epoch: {len(train_dataloader)}")
logger.info(f"Total training steps: {training_steps}")

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=warmup,
    num_training_steps=training_steps,
)

progress_bar = tqdm(range(training_steps))

step_counter = 0
patience_left = patience

best_eval_loss = None
best_eval_bleu = None
best_eval_chr = None

train_loss_log, eval_loss_log, eval_bleu_log, eval_chr_log, lr_log = [], [], [], [], []

scaler = torch.GradScaler()

def train(model, dataloader, optimizer, scheduler):
    # I know, i know, global variables, but it works
    # and the training has to be run only once
    global step_counter
    global eval_dataloader
    global best_eval_loss
    global best_eval_bleu
    global best_eval_chr
    global patience_left
    global eval_loss_log
    global eval_bleu_log
    global eval_chr_log
    total_loss = 0
    model.train()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**batch)
            loss = outputs.loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        progress_bar.update(1)

        loss_value = loss.item()

        total_loss += loss_value

        step_counter += 1

        del loss # Just to make sure, that nothing unnecessary is retained, initally got some OOM errors
        del outputs

        if step_counter % log_steps == 0:
            logger.info(f"Training step: {step_counter}/{len(dataloader)}, loss: {loss_value} lr: {lr_scheduler.get_lr()}")

        if step_counter % eval_steps == 0:
            logger.info(f"Running eval at step {step_counter}")
            logger.info(f"Best eval loss: {best_eval_loss}, eval bleu: {best_eval_bleu}, eval chr: {best_eval_chr}")
            eval_loss, eval_bleu, eval_chr = eval(model, eval_dataloader, tokenizer)
            logger.info(f"Eval loss: {eval_loss}, eval bleu: {eval_bleu}, eval chr: {eval_chr}")

            eval_loss_log.append(eval_loss)
            eval_bleu_log.append(eval_bleu)
            eval_chr_log.append(eval_chr)

            # Uninitialized eval loss
            if best_eval_loss == None:
                best_eval_loss = eval_loss
                best_eval_bleu = eval_bleu
                best_eval_chr = eval_chr
                continue

            if eval_loss < best_eval_loss:
                logger.info(f"Eval loss improved, resetting patience to {patience}")
                patience_left = patience
                best_eval_loss = eval_loss
                best_eval_bleu = eval_bleu
                best_eval_chr = eval_chr

                save_path = f"{output}/last" if output != "" else "last"
                logger.info(f"Saving the checkpoint to {save_path}")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
            else:
                patience_left -= 1
                logger.info(f"Eval loss not improved, continuing training, patience left: {patience_left}")

                if patience_left == 0:
                    logger.info("Patience 0, stopping training early")
                    return -1

    return total_loss / len(dataloader)


def eval(model, dataloader, tokenizer):
    total_loss = 0
    total_bleu = 0
    total_chr = 0
    model.eval()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        predictions = tokenizer.batch_decode(predictions.tolist(), skip_special_tokens=True)
        labels = [[id if id >= 0 else 1 for id in sent] for sent in batch["labels"].tolist()]
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        total_loss += outputs.loss.item()
        total_bleu += sacrebleu.corpus_bleu(predictions, [labels]).score
        total_chr += sacrebleu.corpus_chrf(predictions, [labels], word_order = 2).score

        del outputs

    return total_loss / len(dataloader), total_bleu / len(dataloader), total_chr / len(dataloader)

early_stopping = False
for epoch in range(epochs):
    logger.info(f"Starting epoch: {epoch + 1}")
    train_loss = train(model, train_dataloader, optimizer, lr_scheduler)
    if train_loss == -1:
        early_stopping = True
        break
    train_loss_log.append(train_loss)
    lr_log.append(lr_scheduler.get_lr())

# If the model did not stop due to early stopping, but ran out of steps,
# then do the last evaluation here.
if not early_stopping:
    logger.info("Running last eval to check for better model")
    logger.info(f"Best eval loss: {best_eval_loss}, eval bleu: {best_eval_bleu}, eval chr: {best_eval_chr}")
    eval_loss, eval_bleu, eval_chr = eval(model, eval_dataloader, tokenizer)
    logger.info(f"Eval loss: {eval_loss}, eval bleu: {eval_bleu}, eval chr: {eval_chr}")

    eval_loss_log.append(eval_loss)
    eval_bleu_log.append(eval_bleu)
    eval_chr_log.append(eval_chr)

    if eval_loss < best_eval_loss:
        logger.info(f"Eval loss improved, resetting patience to {patience}")
        patience_left = patience
        best_eval_loss = eval_loss
        best_eval_bleu = eval_bleu
        best_eval_chr = eval_chr

        save_path = f"{output}/last" if output != "" else "last"
        logger.info(f"Saving the checkpoint to {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
else:
    logger.info("Skipping last eval because of early stopping")


if store_metrics:
    logger.info("Storing metrics")
    def save_metric(items, name, output):
        path = f"{output}/{name}" if output != "" else name
        with open(path, "w", encoding="utf-8") as f:
            for metric in items:
                f.write(f"{metric}\n")
        logger.info(f"Stored metric {name} to {path}")

    metrics = [
        (train_loss_log, "train_loss"),
        (eval_loss_log, "eval_loss"),
        (eval_bleu_log, "eval_bleu"),
        (eval_chr_log, "eval_chr"),
        (lr_log, "lr"),
    ]

    for items, name in metrics:
        save_metric(items, name, output)
