"""
Fine-tune Qwen3-ASR on data prepared with prepare_data.py.

    python qwen/finetune.py --data fleurs-fi --quick     # try it: 64 sentences, 30 steps
    python qwen/finetune.py --data fleurs-fi             # full run, 3 epochs

The model is saved in $WORK/models/<name>/final, where <name> is
<model>-<language>, e.g. qwen3-asr-1.7b-fi (with --quick: qwen3-asr-1.7b-fi-quick).
Use the name with evaluate.py and transcribe.py.

Qwen3-ASR expects each training transcript to start with a language tag:
    language Finnish<asr_text>Hyönteiset olivat ensimmäisiä eläimiä.
The prepared data has plain transcripts; the tag is added here from the data's
language column, so the same data works for Whisper and Qwen.

The effective batch size is batch_size x grad_acc x number of GPUs.
Based on the fine-tuning example by the Qwen team (Apache-2.0).
"""
import argparse
import io
import json
import os
import shutil
import math

import librosa
import torch
from datasets import load_from_disk
from huggingface_hub import snapshot_download
from qwen_asr import Qwen3ASRModel
from transformers import GenerationConfig, Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

WORK = os.environ.get("WORK", "/scratch/project_462001302/asr-data")
SAMPLE_RATE = 16000
LANGUAGE_NAMES = {"fi": "Finnish", "en": "English"}   # data language code -> Qwen's name

# Files the qwen_asr loader needs next to the weights; the Trainer does not save all of them
PROCESSOR_FILES = ["config.json", "generation_config.json", "preprocessor_config.json",
                   "processor_config.json", "tokenizer_config.json", "tokenizer.json",
                   "special_tokens_map.json", "chat_template.json", "merges.txt", "vocab.json"]


def load_audio(wav_bytes: bytes):
    audio, _ = librosa.load(io.BytesIO(wav_bytes), sr=SAMPLE_RATE, mono=True)
    return audio


def copy_processor_files(base_dir: str, dst_dir: str):
    """Make a checkpoint loadable with Qwen3ASRModel.from_pretrained (never overwrites)."""
    for name in PROCESSOR_FILES:
        src, dst = os.path.join(base_dir, name), os.path.join(dst_dir, name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)


class MakeCheckpointsLoadable(TrainerCallback):
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            copy_processor_files(self.base_dir, os.path.join(args.output_dir, f"checkpoint-{state.global_step}"))


class Collator:
    """Batch of dataset rows -> model inputs. Only the transcript part is trained on."""

    def __init__(self, processor):
        self.processor = processor
        messages = [{"role": "system", "content": ""},
                    {"role": "user", "content": [{"type": "audio", "audio": None}]}]
        # The prompt before the transcript, the same for every sentence
        self.prefix = processor.apply_chat_template([messages], add_generation_prompt=True, tokenize=False)[0]

    def __call__(self, rows):
        audios = [load_audio(r["audio"]) for r in rows]
        eos = self.processor.tokenizer.eos_token or ""
        targets = [f"language {LANGUAGE_NAMES[r['language']]}<asr_text>{r['text']}" for r in rows]
        full = self.processor(text=[self.prefix + t + eos for t in targets], audio=audios,
                              return_tensors="pt", padding=True, truncation=False)
        prefix = self.processor(text=[self.prefix] * len(rows), audio=audios,
                                return_tensors="pt", padding=True, truncation=False)
        labels = full["input_ids"].clone()
        for i, n in enumerate(prefix["attention_mask"].sum(dim=1).tolist()):
            labels[i, :n] = -100                     # don't train on the prompt
        labels[full["attention_mask"] == 0] = -100   # don't train on padding
        full["labels"] = labels
        return full


class CastInputsTrainer(Trainer):
    """Audio features come as float32; the model runs in bf16."""

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        for k, v in inputs.items():
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(self.model.dtype)
        return inputs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Prepared data name, e.g. fleurs-fi")
    p.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--name", default=None, help="Name of the fine-tuned model (default: <model>-<language>)")
    p.add_argument("--quick", action="store_true",
                   help="Try it quickly: 64 training sentences, 30 steps, 32 validation sentences")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch_size", type=int, default=8, help="Per GPU")
    p.add_argument("--grad_acc", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--save_steps", type=int, default=20, help="Validate and save every N steps")
    p.add_argument("--resume", action="store_true", help="Continue from the last checkpoint")
    p.add_argument("--num_workers", type=int, default=6)
    args = p.parse_args()

    data_dir = os.path.join(WORK, "data", args.data)
    train = load_from_disk(os.path.join(data_dir, "train"))
    valid = load_from_disk(os.path.join(data_dir, "validation"))
    language = train[0]["language"]
    max_steps = -1
    if args.quick:
        train, valid = train.select(range(min(64, len(train)))), valid.select(range(min(32, len(valid))))
        max_steps, args.save_steps, args.grad_acc = 30, 10, 1

    name = args.name or f"{args.model.split('/')[-1].lower()}-{language}" + ("-quick" if args.quick else "")
    out_dir = os.path.join(WORK, "models", name)
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"Model (start) : {args.model}")
        print(f"Data          : {data_dir} (language {language} -> tag 'language {LANGUAGE_NAMES[language]}')")
        print(f"Training      : {len(train)} sentences, {sum(train['duration']) / 3600:.2f} h, "
              + (f"{max_steps} steps" if max_steps > 0 else f"{args.epochs:g} epochs"))
        print(f"Validation    : {len(valid)} sentences (loss), every {args.save_steps} steps")
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

    wrapper = Qwen3ASRModel.from_pretrained(args.model, dtype=torch.bfloat16, device_map=None)
    model, processor = wrapper.model, wrapper.processor
    base_dir = args.model if os.path.isdir(args.model) else snapshot_download(args.model)

    # The Trainer calls model.forward(); Qwen3-ASR keeps the real forward in model.thinker
    def forward(self, input_ids=None, attention_mask=None, input_features=None,
                feature_attention_mask=None, labels=None, **kwargs):
        return self.thinker.forward(input_ids=input_ids, attention_mask=attention_mask,
                                    input_features=input_features,
                                    feature_attention_mask=feature_attention_mask, labels=labels, **kwargs)
    model.__class__.forward = forward
    model.generation_config = GenerationConfig.from_model_config(model.config)

    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        warmup_ratio=0.02,
        bf16=True,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=10,
        report_to="none",
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,   # the collator needs the "audio", "text" and "language" columns
        ddp_find_unused_parameters=False,
    )
    trainer = CastInputsTrainer(model=model, args=training_args, data_collator=Collator(processor),
                                train_dataset=train, eval_dataset=valid, processing_class=processor,
                                callbacks=[MakeCheckpointsLoadable(base_dir)])

    resume = get_last_checkpoint(out_dir) if args.resume and os.path.isdir(out_dir) else None
    trainer.train(resume_from_checkpoint=resume)

    final_dir = os.path.join(out_dir, "final")
    trainer.save_model(final_dir)            # the model after the last step
    if trainer.is_world_process_zero():
        copy_processor_files(base_dir, final_dir)
        with open(os.path.join(final_dir, "training_summary.json"), "w") as f:
            json.dump({"args": vars(args), "start_model": args.model,
                       "history": [h for h in trainer.state.log_history if "eval_loss" in h]}, f, indent=2)
        print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB of 64 GB")
        print(f"\nModel saved in {final_dir}")
        print(f"Use it with:  --model {name}")


if __name__ == "__main__":
    main()
