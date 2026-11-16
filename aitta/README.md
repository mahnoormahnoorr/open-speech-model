# Speech-to-text through the AITTA API

Send audio to a Whisper model running on [CSC's **AITTA** service](https://aitta.csc.fi), through its
OpenAI-compatible API, and compare the transcripts with the correct texts. The
model runs on AITTA's GPUs, so you need no GPU of your own.

| File | What it does |
|---|---|
| `aitta_asr.ipynb` | Notebook: gets the data, transcribes one recording, then up to **50**, then explains and calculates WER |
| `transcribe_all.py` + `transcribe_all.sh` | Batch job: **all** recordings of a split (about 900), on a CPU node |

Start with the notebook, then use the batch job for everything. The notebook
stops at 50 recordings on purpose: a notebook stops when the Jupyter session
ends, a batch job doesn't.

## What you need

- **An AITTA access token.** [PLACEHOLDER: how to get access to AITTA and a token]
  Keep it secret: don't write it into a notebook cell or a script. The notebook
  asks for it, or reads it from the environment variable `AITTA_API_KEY`.
- **Nothing else.** The notebook downloads the data it needs (the FLEURS test
  split, a few hundred MB) the first time it runs, and saves it in
  `/scratch/project_462001302/asr-data/data/fleurs-fi/`. If the data is already
  there, for example from `sbatch prepare_data.sh --language fi` in the
  repository root, it is used as it is.

## The notebook in Jupyter on LUMI

1. Go to [www.lumi.csc.fi](https://www.lumi.csc.fi), log in and open the **Jupyter** app.
2. Fill in the form:

   | Field | Choose |
   |---|---|
   | Project | `project_462001302` |
   | Partition | `small` (no GPU needed) |
   | CPU cores and memory | 2 cores, 8 GB |
   | Time | e.g. 2 hours |
   | Working directory | this `aitta` folder |
   | Python | **lumi-multitorch** |

3. Click **Launch**, then **Connect to Jupyter**, open `aitta_asr.ipynb` and run
   the cells from the top.

When you're done, **delete the session** on the *My Interactive Sessions* page.
It uses billing units for as long as it runs.

## The batch job: all recordings

From the repository root. Run the notebook's data section first (or
`sbatch prepare_data.sh --language fi`): the batch job uses the data already
on `/scratch` and does not download it.

```bash
export AITTA_API_KEY=<your token>           # the job gets the token from here
sbatch aitta/transcribe_all.sh --data fleurs-fi
```

One recording is one API call, sent one after another. Each result is saved as
soon as it arrives, so if the job stops, or some calls fail, just submit it
again: recordings already done are skipped. The job also continues from what
the notebook already saved, because both use the same results folder.

**Look at:** the WER/CER table at the end of the log, and the number of failed
calls if any.

## Where the results go

`/scratch/project_462001302/asr-data/results/aitta-whisper-large-v3-fleurs-fi-test/`:

| File | Contents |
|---|---|
| `transcripts.jsonl` | One line per recording, saved as soon as it is done |
| `side_by_side.tsv` | Correct text, transcript and WER per sentence; opens in Excel or LibreOffice |
| `summary.md` | WER and CER over all sentences (written by the batch job) |

These are the same files as from `whisper/evaluate.sh` and `qwen/evaluate.sh`,
so the results can be compared directly.

## Good to know

- **Data protection.** The audio is sent to an external service. That's fine for
  public data like FLEURS, but check AITTA's terms before sending your own or
  sensitive recordings.

| Problem | What to do |
|---|---|
| `ModuleNotFoundError` in the notebook | Choose **lumi-multitorch** as the Python in the Jupyter form. |
| `401` or `Unauthorized` | The access token is wrong or expired. |
| `429` or `Too Many Requests` | Too many calls in a short time. The client retries by itself; if calls still fail, wait a moment and run again. |
| `ERROR: set AITTA_API_KEY` in the job log | Run `export AITTA_API_KEY=...` in the same terminal before `sbatch`. |
