"""
Fine-tune a Whisper model on data prepared by prepare_data.py.
Started by run-finetune-lumi-gpu1.sh / run-finetune-lumi-gpu8.sh, which choose
the Whisper model (openai/whisper-large-v3-turbo unless you give --model).

    python finetune_whisper.py --model openai/whisper-small --data fleurs-fi

The fine-tuned model is saved as $OUTPUT_DIR/models/<name>/final, where
<name> defaults to <model>-<language>, e.g. whisper-small-fi. Use that name
with compare.py and inference.py.

Other examples:
    # Try it quickly: first 64 sentences, 30 steps -> model whisper-small-fi-quick
    python finetune_whisper.py --model openai/whisper-small --data fleurs-fi --quick

    # Overfit check: train and validate on the same 32 sentences; WER should drop sharply
    python finetune_whisper.py --model openai/whisper-small --data fleurs-fi \
        --overfit 32 --max_steps 100 --eval_steps 25 --learning_rate 1e-4

The effective batch size is batch_size x grad_acc x number of GPUs.
"""
import argparse
import json
import os
import sys

import torch
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperForConditionalGeneration, WhisperProcessor)
from transformers.trainer_utils import get_last_checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asr_common import (LANGUAGES, SAMPLE_RATE, data_language, decode_audio,  # noqa: E402
                        QUICK_SUFFIX, load_prepared_split, model_label, models_root,
                        print_setup, resolve_model, wer_cer)

MAX_LABEL_TOKENS = 448  # Whisper decoder limit


class WhisperCollator:
    """Decodes audio, computes log-mel features and builds the label sequence.

    Labels are: <|lang|> <|transcribe|> <|notimestamps|> text <|endoftext|>.
    <|startoftranscript|> is NOT included: the model prepends it itself
    (decoder_start_token_id) when shifting labels right.
    """

    def __init__(self, processor, language: str):
        self.fe = processor.feature_extractor
        self.tok = processor.tokenizer
        whisper_lang = LANGUAGES[language]["whisper"]
        lang_code = next(code for code, name in _whisper_languages().items() if name == whisper_lang)
        self.prefix = self.tok.convert_tokens_to_ids(
            [f"<|{lang_code}|>", "<|transcribe|>", "<|notimestamps|>"])
        self.eos = self.tok.convert_tokens_to_ids("<|endoftext|>")
        assert None not in self.prefix and self.tok.unk_token_id not in self.prefix, \
            "Whisper special tokens not found in tokenizer"

    def __call__(self, batch):
        audios = [decode_audio(data=ex["audio"]) for ex in batch]
        features = self.fe(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        label_seqs = []
        for ex in batch:
            text_ids = self.tok(ex["text"], add_special_tokens=False).input_ids
            label_seqs.append((self.prefix + text_ids)[:MAX_LABEL_TOKENS - 1] + [self.eos])
        max_len = max(len(s) for s in label_seqs)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, s in enumerate(label_seqs):
            labels[i, :len(s)] = torch.tensor(s)
        return {"input_features": features, "labels": labels}


def _whisper_languages():
    from transformers.models.whisper.tokenization_whisper import LANGUAGES as WHISPER_LANGS
    return WHISPER_LANGS


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="openai/whisper-small",
                   help="Starting model: Hugging Face id or one of your fine-tuned models")
    p.add_argument("--data", "--data_dir", dest="data", required=True,
                   help="Prepared data name (e.g. fleurs-fi) or path")
    p.add_argument("--language", default=None, choices=sorted(LANGUAGES),
                   help="Default: the language stored in the data")
    p.add_argument("--name", default=None,
                   help="Name of the fine-tuned model (default: <model>-<language>)")
    p.add_argument("--output_dir", default=None, help="Default: $OUTPUT_DIR/models/<name>")
    p.add_argument("--quick", action="store_true",
                   help="Try it quickly: first 64 training sentences, 30 steps, validation on "
                        "32 sentences; the model name gets '-quick'")
    # Training length
    p.add_argument("--epochs", type=float, default=5)
    p.add_argument("--max_steps", type=int, default=None,
                   help="Overrides --epochs (default: none; 30 with --quick)")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--overfit", type=int, default=None, metavar="N",
                   help="Sanity check: train AND evaluate on the first N training examples")
    # Optimization
    p.add_argument("--batch_size", type=int, default=16, help="Per GPU")
    p.add_argument("--grad_acc", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--gradient_checkpointing", action="store_true")
    # Evaluation / checkpoints
    p.add_argument("--eval_steps", type=int, default=None,
                   help="Also the checkpoint interval (default 100; 10 with --quick)")
    p.add_argument("--max_eval_samples", type=int, default=None,
                   help="Validation sentences used during training (default 200; 32 with --quick)")
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--resume", action="store_true", help="Continue from the last checkpoint")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--report_to", default="none", help="e.g. tensorboard, mlflow, wandb")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.quick and args.max_train_samples is None and not args.overfit:
        args.max_train_samples = 64
    if args.max_steps is None:
        args.max_steps = 30 if args.quick else -1
    if args.eval_steps is None:
        args.eval_steps = 10 if args.quick else 100
    if args.max_eval_samples is None:
        args.max_eval_samples = 32 if args.quick else 200
    args.language = args.language or data_language(args.data)
    args.model = resolve_model(args.model)
    name = args.name or (f"{model_label(args.model)}-{args.language}"
                         + (QUICK_SUFFIX if args.quick else ""))
    output_dir = args.output_dir or os.path.join(models_root(), name)

    # ---- Data ---------------------------------------------------------------
    train_ds = load_prepared_split(args.data, "train")
    if args.overfit:
        train_ds = train_ds.select(range(min(args.overfit, len(train_ds))))
        eval_ds = train_ds
    else:
        if args.max_train_samples:
            train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
        eval_ds = load_prepared_split(args.data, "validation")
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))
    print_setup("Fine-tuning setup", {
        "Model (start)": args.model,
        "Data": f"{args.data} (language: {args.language})",
        "Training data": f"{len(train_ds)} sentences, {sum(train_ds['duration']) / 3600:.2f} h",
        "Validation data": f"{len(eval_ds)} sentences"
                           + (" (= the training sentences, overfit check)" if args.overfit else ""),
        "Training length": f"{args.max_steps} steps" if args.max_steps > 0 else f"{args.epochs:g} epochs",
        "Batch per GPU": f"{args.batch_size} (gradient accumulation {args.grad_acc})",
        "Learning rate": args.learning_rate,
        "Validate/save every": f"{args.eval_steps} steps",
        "Output": output_dir,
        "Model name": name if not args.output_dir else "(custom --output_dir)",
        "Mode": "QUICK (for trying things out)" if args.quick else "full",
    })

    # ---- Model --------------------------------------------------------------
    processor = WhisperProcessor.from_pretrained(args.model)
    # Load in fp32: bf16=True below then gives proper mixed precision
    # (fp32 master weights), not pure bf16 training.
    model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32)
    model.generation_config.language = LANGUAGES[args.language]["whisper"]
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    if getattr(model.config, "forced_decoder_ids", None) is not None:
        model.config.forced_decoder_ids = None
    if args.gradient_checkpointing:
        model.config.use_cache = False

    collator = WhisperCollator(processor, args.language)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
        hyps = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        refs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        m = wer_cer(refs, hyps)
        return {"wer": m["wer"], "cer": m["cer"]}

    # ---- Training arguments -------------------------------------------------
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    warmup = args.warmup_steps
    if args.max_steps > 0:
        warmup = min(warmup, max(1, args.max_steps // 10))
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.learning_rate,
        warmup_steps=warmup,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        bf16=use_bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=max(1, min(10, args.eval_steps)),
        report_to=args.report_to,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,   # the collator needs the raw "audio"/"text" columns
        ddp_find_unused_parameters=False,
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=training_args, data_collator=collator,
        train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=processor, compute_metrics=compute_metrics,
    )

    world = training_args.world_size
    steps_per_epoch = len(train_ds) / (args.batch_size * args.grad_acc * world)
    if trainer.is_world_process_zero():
        print(f"GPUs: {world}, effective batch size: {args.batch_size * args.grad_acc * world}, "
              f"~{steps_per_epoch:.1f} training steps per epoch, bf16 mixed precision: {use_bf16}")

    resume = get_last_checkpoint(output_dir) if args.resume and os.path.isdir(output_dir) else None
    if resume:
        print(f"Resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)

    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)          # best checkpoint (by validation WER)
    if trainer.is_world_process_zero():
        processor.save_pretrained(final_dir)
        history = [h for h in trainer.state.log_history if "eval_wer" in h]
        with open(os.path.join(final_dir, "training_summary.json"), "w") as f:
            json.dump({"args": vars(args), "best_checkpoint": trainer.state.best_model_checkpoint,
                       "best_eval_wer": trainer.state.best_metric, "eval_history": history},
                      f, indent=2)
        print(f"\nBest validation WER: {trainer.state.best_metric}")
        print(f"FINAL MODEL: {final_dir}")
        if not args.output_dir:
            print(f"Use it with: --model {name}")


if __name__ == "__main__":
    main()