# Fine-tuning Qwen3-ASR for Finnish speech data on LUMI 
 
This repo documents fine-tuning [Qwen3-ASR-1.7B](https://github.com/QwenLM/Qwen3-ASR) on Finnish speech data using [LUMI](https://www.lumi-supercomputer.eu/), via the LUMI AI Factory container stack.
 
## Overview
 
- **Base model:** `Qwen/Qwen3-ASR-1.7B`
- **Hardware:** LUMI `dev-g` partition, 8x AMD MI250x GCDs per node
- **Data:** Google FLEURS Finnish subset (`fi_fi`) — ~2,700 train / ~415 validation examples, ~8.8 hours of audio
- **Container:** LUMI AI Factory `lumi-multitorch-full` image (PyTorch + ROCm 7.0 + DeepSpeed + vLLM bundled)

## Setup
 
```bash
module load Local-LAIF lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/<lumi-multitorch-full-image>.sif
 
singularity shell "$SIF"
python -m venv venv --system-site-packages
source venv/bin/activate
pip install "transformers==4.57.6" "qwen-asr==0.0.6" datasets
```

## Prepare your training data 

Put your training data in a JSONL file (for example `train.jsonl`), with one JSON object per line. Each line describes one audio clip and needs two fields:

`audio`: the path to your WAV file
`text`: the transcript, starting with a language tag

Your file should look like this:


```bash
{"audio": "data/audio/train/0.wav", "text": "language Finnish<asr_text>Hyönteiset olivat ensimmäisiä eläimiä."}
{"audio": "data/audio/train/1.wav", "text": "language Finnish<asr_text>Sää on tänään aurinkoinen."}
{"audio": "data/audio/train/2.wav", "text": "language Finnish<asr_text>Tämä on harjoitusesimerkki."}
```

### Add the language tag

Start every transcript with `language <Lang><asr_text>`. Don’t skip this: `Qwen3-ASR` uses this marker to know where the transcription begins.

If you know the language of the clip, write its name in the tag, e.g. `language Finnish<asr_text>...` or `language English<asr_text>...`
If you don’t know it, write `language None<asr_text>...`

Tip: Clips tagged `None` don’t teach the model anything about language detection. If you want your fine-tuned model to recognise the language correctly, label as many clips as you can.


`prep_audio.py` downloads FLEURS Finnish and writes one wav file per utterance plus a jsonl manifest. The corresponding write block in `prep_audio.py`: 

```bash
f.write(json.dumps({
    "audio": os.path.abspath(wav_path),
    "text": f"language Finnish<asr_text>{text}",
}, ensure_ascii=False) + "\n")
```

Run it once per split:

```bash
python prep_audio.py --split train --out_dir data
python prep_audio.py --split validation --out_dir data
python prep_audio.py --split test  --out_dir data
```

Result: 2,704 train, 415 validation and 918 test utterances. The splits are disjoint. Training uses only `train`, `validation` is used for eval_loss during training, and `test` is kept for the final evaluation.

### Fine-tune (single GPU) 
 
```bash
sbatch run-finetuning-lumi-gpu1.sh
```


### Fine-tune (multi GPU with torch run) 

```bash
sbatch run-finetuning-lumi-gpu8.sh
```

### Resume training 

Training can be resumed from checkpoints if required. Add argument `--resume 1` to automatically resume from latest checkpoint or `--resume_from ./qwen3-asr-finetuning-out/checkpoint-200` to resume from a checkpoint path.

### Run inference
 
```bash
sbatch run-inference.sh
```
 

