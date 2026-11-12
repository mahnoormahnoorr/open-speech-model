# Speech-to-text through the Aitta API: notebooks

These notebooks send audio to a Whisper speech-to-text model running on CSC's
**Aitta** service, through its **OpenAI-compatible API**, and compare the
model's transcripts with the correct texts.

The model runs on Aitta's GPUs, not on your computer. The notebook only reads
the audio, sends it, and handles the results, so it needs no GPU of its own: you
can run it **in Jupyter on LUMI** or **on your own computer**.

## The notebooks

Go through them in order:

| Notebook | What it does |
|---|---|
| `1_first_api_call.ipynb` | Transcribes **one recording at a time**: look at and listen to a recording, send it to the API, and see the model's transcript next to the correct text. Also shows how to send your own audio file. |
| `2_transcribe_many.ipynb` | Transcribes **many recordings**, up to a whole part of the dataset, saving each result as it arrives (so a run can be stopped and continued). Then explains how the word error rate (WER) is calculated, scores the transcripts, and shows the sentences with the most errors. |

## What you need

- **An Aitta access token.** [PLACEHOLDER: how to get access to Aitta and a token]
  The notebooks ask for it when they start, or read it from the environment
  variable `AITTA_API_KEY`. Keep it secret: don't write it into a notebook cell.
- **Prepared data.** The notebooks use FLEURS data prepared with the course
  scripts (`sbatch run-prepare-data-lumi.sh --language fi`, see the course
  repository's README). On your own computer, see step 3 under
  [Option B](#option-b-on-your-own-computer).

The notebooks are self-contained: everything else they need, including the code
that calculates WER, is in the notebooks themselves. The WER code is the same as
in the course repository's `asr_common.py`, so the results are directly
comparable with the course's comparison script.

## Option A: Jupyter on LUMI

LUMI's web interface can start JupyterLab on a LUMI compute node, which you then
use in your browser.

1. Go to [www.lumi.csc.fi](https://www.lumi.csc.fi) and log in.
2. Open the **Jupyter** app (under *Pinned apps*, or *Apps* in the top bar).
3. Fill in the form:

   | Field | Choose |
   |---|---|
   | Project | Your LUMI project |
   | Partition | A CPU partition, e.g. `small`. **No GPU is needed**, since the model runs on Aitta. |
   | CPU cores and memory | Small is enough, e.g. 2 cores and 8 GB |
   | Time | How long you'll work, e.g. 2 hours |
   | Working directory | This folder, e.g. `/projappl/<project>/<username>/asr-lumi/notebooks` [PLACEHOLDER: check the folder name in the repository] |
   | Settings | **Basic** |
   | Python | **lumi-multitorch** (the LUMI AI Factory environment; it has all the packages the notebooks use) |

4. Click **Launch**, wait until the session is running, and click **Connect to Jupyter**.
5. Open `1_first_api_call.ipynb` and run the cells from the top.

On LUMI the notebooks find your prepared data automatically under
`/scratch/<project>/<username>/asr/`, so you don't need to change any paths.

The session uses your project's CPU billing units for as long as it runs. When
you're done, **delete the session** on the *My Interactive Sessions* page instead
of just closing the browser tab.

More about the Jupyter app: [LUMI documentation](https://docs.lumi-supercomputer.eu/runjobs/webui/jupyter/)

## Option B: on your own computer

Because the GPU work happens on Aitta, the notebooks also run on an ordinary
laptop with an internet connection.

1. **Install Python** 3.10 or newer, then the packages, preferably in a virtual
   environment:
   ```bash
   python3 -m venv asr-env
   source asr-env/bin/activate          # Windows: asr-env\Scripts\activate
   pip install jupyterlab datasets openai pandas
   ```
2. **Get the notebooks**: download the two `.ipynb` files from this folder, or
   clone the whole course repository.
3. **Get the data.** Either copy the prepared test data from LUMI (about
   300 MB for Finnish):
   ```bash
   mkdir -p ~/asr/data/fleurs-fi
   scp -r <username>@lumi.csc.fi:/scratch/<project>/<username>/asr/data/fleurs-fi/test ~/asr/data/fleurs-fi/
   ```
   or prepare it on your own computer with `prepare_data.py` from the course
   repository [PLACEHOLDER: repository URL] (this downloads about 2 GB):
   ```bash
   pip install huggingface_hub scipy
   python prepare_data.py --language fi --output_dir ~/asr/data/fleurs-fi
   ```
4. **Start JupyterLab** in the folder with the notebooks. Setting the token first means
   you don't have to type it in the notebook:
   ```bash
   export AITTA_API_KEY=<your token>    # optional
   jupyter lab
   ```
5. **Tell the notebooks where the data is.** In section 2 of each notebook, set
   ```python
   WORK_DIR = "~/asr"
   ```
   (the folder that contains `data/fleurs-fi/test`). Results are then saved under
   `~/asr/runs/api/`.

## Where the results go

Notebook 2 saves its results in `<WORK_DIR>/runs/api/<run name>/`, on LUMI
`/scratch/<project>/<username>/asr/runs/api/...`:

| File | Contents |
|---|---|
| `transcriptions.jsonl` | One line per recording, saved as soon as it is done |
| `side_by_side.tsv` | Correct text, transcript and WER per sentence; opens in Excel or LibreOffice |

## Good to know

- **One recording is one API call.** Transcribing the whole Finnish test set
  means 916 calls, sent one after another, which takes a while. Start with a few
  recordings (`MAX_ENTRIES`), and check Aitta's usage limits.
- **Data protection.** The audio is sent to an external service. That's fine for
  public data like FLEURS, but check Aitta's terms before sending your own or
  sensitive recordings.

| Problem | What to do |
|---|---|
| `ModuleNotFoundError` | On LUMI, choose **lumi-multitorch** as the Python in the Jupyter form. Locally, install the packages from step 1. |
| `No /scratch/<project>/<user>/asr folder found` | Prepare the data first, or on your own computer set `WORK_DIR` (step 5). |
| `401` or `Unauthorized` | The access token is wrong or expired. |
| `429` or `Too Many Requests` | Too many calls in a short time. The client retries a few times by itself; if calls still fail, wait a moment and run section 5 of notebook 2 again. |
