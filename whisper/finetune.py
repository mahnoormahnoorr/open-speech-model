"""
Fine-tune a Whisper model on data prepared with prepare_data.py.

    python whisper/finetune.py --data fleurs-fi --quick     # try it: whisper-small, 64 sentences, 30 steps
    python whisper/finetune.py --data fleurs-fi             # full: whisper-large-v3-turbo, 5 epochs
    python whisper/finetune.py --data fleurs-fi --model openai/whisper-small

The model is saved in $WORK/models/<name>/final, where <name> is
<model>-<language>, e.g. whisper-small-fi (with --quick: whisper-small-fi-quick).
The checkpoint with the best validation WER is kept as the final model.
Use the name with evaluate.py and transcribe.py.

The effective batch size is batch_size x grad_acc x number of GPUs.
"""
import argparse
import io
import json
import os
import math

import jiwer
import librosa
import torch
from datasets import load_from_disk
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperForConditionalGeneration, WhisperProcessor)
from transformers.trainer_utils import get_last_checkpoint

WORK = os.environ.get("WORK", "/scratch/project_462001302/asr-data")
SAMPLE_RATE = 16000


def normalize(text: str) -> str:
    """For WER: lowercase, punctuation removed, spaces collapsed (same in every script)."""
    text = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
    return " ".join(text.split())


def load_audio(wav_bytes: bytes):
    audio, _ = librosa.load(io.BytesIO(wav_bytes), sr=SAMPLE_RATE, mono=True)
    return audio


class Collator:
    """Turns a batch of dataset rows into model inputs.

    Labels: <|fi|> <|transcribe|> <|notimestamps|> text <|endoftext|>. The model
    adds <|startoftranscript|> itself, so it is not in the labels.
    """

    def __init__(self, processor, language: str):
        self.processor = processor
        tok = processor.tokenizer
        self.prefix = tok.convert_tokens_to_ids([f"<|{language}|>", "<|transcribe|>", "<|notimestamps|>"])
        self.eos = tok.eos_token_id

    def __call__(self, rows):
        audios = [load_audio(r["audio"]) for r in rows]
        features = self.processor.feature_extractor(
            audios, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        label_ids = [(self.prefix + self.processor.tokenizer(r["text"], add_special_tokens=False).input_ids)[:447]
                     + [self.eos] for r in rows]
        labels = torch.full((len(rows), max(map(len, label_ids))), -100, dtype=torch.long)
        for i, ids in enumerate(label_ids):
            labels[i, :len(ids)] = torch.tensor(ids)
        return {"input_features": features, "labels": labels}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Prepared data name, e.g. fleurs-fi")
    p.add_argument("--model", default=None,
                   help="Default: openai/whisper-large-v3-turbo (openai/whisper-small with --quick)")
    p.add_argument("--name", default=None, help="Name of the fine-tuned model (default: <model>-<language>)")
    p.add_argument("--quick", action="store_true",
                   help="Try it quickly: 64 training sentences, 30 steps, 32 validation sentences")
    p.add_argument("--epochs", type=float, default=5)
    p.add_argument("--batch_size", type=int, default=16, help="Per GPU")
    p.add_argument("--grad_acc", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--eval_steps", type=int, default=100, help="Validate and save every N steps")
    p.add_argument("--resume", action="store_true", help="Continue from the last checkpoint")
    p.add_argument("--num_workers", type=int, default=6)
    args = p.parse_args()

    model_id = args.model or ("openai/whisper-small" if args.quick else "openai/whisper-large-v3-turbo")
    local = os.path.join(WORK, "models", model_id, "final")   # continue from your own fine-tuned model?
    model_path = local if os.path.isdir(local) else model_id

    data_dir = os.path.join(WORK, "data", args.data)
    train = load_from_disk(os.path.join(data_dir, "train"))
    valid = load_from_disk(os.path.join(data_dir, "validation"))
    language = train[0]["language"]
    max_steps = -1
    if args.quick:
        train, valid = train.select(range(min(64, len(train)))), valid.select(range(min(32, len(valid))))
        max_steps, args.eval_steps = 30, 10
    else:
        valid = valid.select(range(min(200, len(valid))))   # enough for following progress

    name = args.name or f"{model_id.split('/')[-1]}-{language}" + ("-quick" if args.quick else "")
    out_dir = os.path.join(WORK, "models", name)

    if int(os.environ.get("RANK", 0)) == 0:
        print(f"Model (start) : {model_path}")
        print(f"Data          : {data_dir} (language {language})")
        print(f"Training      : {len(train)} sentences, {sum(train['duration']) / 3600:.2f} h, "
              + (f"{max_steps} steps" if max_steps > 0 else f"{args.epochs:g} epochs"))
        print(f"Validation    : {len(valid)} sentences, every {args.eval_steps} steps")
        print(f"Output        : {out_dir}", flush=True)
       
        # ---- added: training settings ----
        gpus = int(os.environ.get("WORLD_SIZE", 1))
        effective = args.batch_size * args.grad_acc * gpus
        steps_per_epoch = math.ceil(len(train) / effective)
        total_steps = max_steps if max_steps > 0 else math.ceil(steps_per_epoch * args.epochs)
        print(f"GPUs          : {gpus}")
        print(f"Batch         : {args.batch_size} per GPU x grad_acc {args.grad_acc} x {gpus} GPUs "
              f"= {effective} sentences per step")
        print(f"Steps         : about {steps_per_epoch} per epoch, {total_steps} in total")
        print(f"Learning rate : {args.lr}, warmup {3 if args.quick else 50} steps")
        print(f"Precision     : {'bf16 mixed precision' if torch.cuda.is_available() else 'fp32 (no GPU)'}", flush=True)

    processor = WhisperProcessor.from_pretrained(model_path)
    # fp32 weights + bf16=True below = mixed precision training
    model = WhisperForConditionalGeneration.from_pretrained(model_path, dtype=torch.float32)
    model.generation_config.language = language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    def compute_metrics(pred):
        pred_ids, label_ids = pred.predictions, pred.label_ids
        pad = processor.tokenizer.pad_token_id
        pred_ids[pred_ids == -100] = pad
        label_ids[label_ids == -100] = pad
        hyps = [normalize(t) for t in processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
        refs = [normalize(t) for t in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]
        return {"wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        warmup_steps=3 if args.quick else 50,
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        bf16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=10,
        report_to="none",
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,   # the collator needs the "audio" and "text" columns
        ddp_find_unused_parameters=False,
    )
    trainer = Seq2SeqTrainer(model=model, args=training_args, data_collator=Collator(processor, language),
                             train_dataset=train, eval_dataset=valid,
                             processing_class=processor, compute_metrics=compute_metrics)

    resume = get_last_checkpoint(out_dir) if args.resume and os.path.isdir(out_dir) else None
    trainer.train(resume_from_checkpoint=resume)

    final_dir = os.path.join(out_dir, "final")
    trainer.save_model(final_dir)            # the best checkpoint (lowest validation WER)
    if trainer.is_world_process_zero():
        processor.save_pretrained(final_dir)
        with open(os.path.join(final_dir, "training_summary.json"), "w") as f:
            json.dump({"args": vars(args), "start_model": model_path,
                       "best_validation_wer": trainer.state.best_metric,
                       "history": [h for h in trainer.state.log_history if "eval_wer" in h]}, f, indent=2)
        print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB of 64 GB") # Added
        print(f"\nBest validation WER: {trainer.state.best_metric}")
        print(f"Model saved in {final_dir}")
        print(f"Use it with:  --model {name}")


if __name__ == "__main__":
    main()
