# Fine-tuning Qwen3-ASR for Finnish speech data on LUMI 
 
This repo documents fine-tuning [Qwen3-ASR-1.7B](https://github.com/QwenLM/Qwen3-ASR) on Finnish speech data using [LUMI](https://www.lumi-supercomputer.eu/), a EuroHPC supercomputer with AMD MI250x GPUs (ROCm), via the LUMI AI Factory container stack.
 
## Overview
 
- **Base model:** `Qwen/Qwen3-ASR-1.7B`
- **Hardware:** LUMI `dev-g` partition, 8x AMD MI250x GCDs per node
- **Data:** Google FLEURS Finnish subset (`fi_fi`) — ~2,700 train / ~415 validation examples, ~8.8 hours of audio
- **Container:** LUMI AI Factory `lumi-multitorch-full` image (PyTorch + ROCm 7.0 + DeepSpeed + vLLM bundled)
## Files
 
| File | Purpose |
|------|---------|
| `prep_audio.py` | Downloads FLEURS Finnish data from Hugging Face, decodes audio, writes `data/{train,validation}.jsonl` + `.wav` files |
| `finetuning.py` | Main fine-tuning script — custom data collator, prefix/target masking, ROCm-adapted bf16 detection |
| `inference-demo.py` | Runs transcription on validation audio using a given checkpoint, prints predicted vs. gold text |
| `run-finetuning-lumi-gpu8.sh` | SLURM script: launches fine-tuning across 8 GPUs via `torchrun` |
| `run-inference-lumi.sh` | SLURM script: launches inference test on 1 GPU |
 
## Setup
 
```bash
module load Local-LAIF lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/<lumi-multitorch-full-image>.sif
 
singularity shell "$SIF"
python -m venv venv --system-site-packages
source venv/bin/activate
pip install "transformers==4.57.6" "qwen-asr==0.0.6" datasets
```
 
### Data prep
 
```bash
python prep_audio.py --split train --out_dir data
python prep_audio.py --split validation --out_dir data
```
 
### Fine-tune
 
```bash
sbatch lumi.sh
```
 
### Run inference on a checkpoint
 
```bash
sbatch infer_lumi.sh
```
 
## Data format
 
Each line in `train.jsonl` / `validation.jsonl`:
 
```json
{"audio": "data/audio/train/0.wav", "text": "Hyönteiset olivat ensimmäisiä eläimiä.", "prompt": "Transcribe the following Finnish audio."}
```
 
The fine-tuning script transforms the target text into Qwen3-ASR's expected training format before computing loss:
 
```
language Finnish<asr_text>Hyönteiset olivat ensimmäisiä eläimiä.
```
 
This marker (`language <Lang><asr_text>`) is required per the official fine-tuning spec — it's the boundary token sequence the pretrained model uses to know the prompt has ended and transcription should begin.
 
## Known issues encountered on LUMI / ROCm
 
- **`qwen-asr` requires `transformers==4.57.6` exactly.** The LAIF container ships a newer `transformers` (5.x) system-wide; a venv built with `--system-site-packages` can silently shadow your pinned version unless `PYTHONPATH` explicitly prioritizes the venv's own `site-packages` over `/opt/venv/lib/.../site-packages`.
- **Apptainer's `--cleanenv`-style wrapping** (via the `lumi-aif-singularity-bindings` module) means plain `export PYTHONPATH=...` on the host does not reliably reach the container. `APPTAINERENV_PYTHONPATH=...` or `singularity run --env PYTHONPATH=...` (properly quoted) are the mechanisms that actually work.
- **Do not use `--rocm`** with this specific container — it bundles its own self-contained ROCm 7.0 userspace; binding the host's older ROCm 6.3.4 libraries on top causes `undefined symbol` crashes. The `lumi-aif-singularity-bindings` module already handles device access correctly without it.
- **`qwen_asr`'s bundled `modeling_qwen3_asr.py`** uses `@check_model_inputs()` (parens) which breaks under some `transformers` versions expecting the non-factory form (`@check_model_inputs`, no parens) — a one-line `sed` patch fixes this if it recurs.
- **`Qwen3ASRModel.from_pretrained(...)`'s `thinker_config` `AttributeError`** appears whenever `transformers` drifts away from `4.57.6` — always re-check `python -c "import transformers; print(transformers.__version__)"` if this resurfaces.
- **Checkpoints saved mid-training are missing files** (`preprocessor_config.json`, `chat_template.json`, etc.) if `--model_path` was passed as a bare HF repo id (e.g. `Qwen/Qwen3-ASR-1.7B`) rather than a resolved local path — the checkpoint-fixing callback's `os.path.join` silently fails to find files to copy. Workaround: copy the missing files manually from the resolved HF cache snapshot (`$HF_HOME/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/<hash>/`) before running inference.
