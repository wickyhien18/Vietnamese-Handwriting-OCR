# Vietnamese Handwriting OCR + Diacritic Correction

A three-person machine learning project.

```
handwritten line image --> OCR (role A) --> diacritic corrector (role B) --> corrected text
                                   role C: evaluation, integration, demo
```

Research question: how much does a diacritic-correction step reduce OCR errors on handwritten Vietnamese, and when does it make things worse (changing a word that was already right)?

See also: [`docs/ASSIGNMENTS.md`](docs/ASSIGNMENTS.md) (who does what and when) and [`docs/SETUP_CHECKLIST.md`](docs/SETUP_CHECKLIST.md) (tick-box version of the setup below).

---

## Start here: full setup, in order

Do the steps in order, from Step 0 to Step 7. Every step ends with **"You should see"**, so you know it worked before you move on. If a step fails, look at the troubleshooting table at the end, then send the exact error text to the group.

> **Before you start, the maintainer (Hiển) must have done three things:**
> 1. Invited you as a collaborator on the GitHub repository.
> 2. Shared the Google Drive folder `Vietnamese-Handwriting-OCR` with you as **Editor**.
> 3. Sent you the read-only GitHub token in a private message (you need it in Step 7).

### Step 0. Check that you have access

1. **GitHub invitation.** Open https://github.com/wickyhien18/Vietnamese-Handwriting-OCR/invitations while logged in to GitHub (or use the invitation email) and click **Accept**.
   You should see: the repository page, not a "404" page.
2. **Drive folder.** Open https://drive.google.com and go to **Shared with me**.
   You should see: a folder named `Vietnamese-Handwriting-OCR`.
3. **Token.** You received a long text starting with `github_pat_` from Hiển. Keep it for Step 7. Never post it in a public place and never commit it.

### Step 1. Install Python 3.10+, Git and GitHub CLI

First check what you already have:

```
python3 --version    # Linux / macOS (on Windows use: py --version)
git --version
gh --version
```

If Python is 3.10 or newer and the other two print a version, skip to Step 2. The project was tested on Python 3.12. Older than 3.10 is untested.

**Ubuntu, Debian, Linux Mint, Pop!\_OS**

```
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git gh
```

`python3-venv` is a separate package on these systems. Without it, creating a virtual environment fails. If apt says it cannot find `gh`, follow https://cli.github.com for the extra package source.

**CachyOS, Arch Linux, Manjaro, EndeavourOS** (Arch-based, uses `pacman`)

```
sudo pacman -Syu
sudo pacman -S python git github-cli
```

On Arch-based systems the `python` package normally includes `venv`. If creating the environment later fails with an `ensurepip` error, also run `sudo pacman -S python-pip`. CachyOS uses the **fish** shell by default, so use the fish activation command in Step 4.

**Fedora, RHEL, Rocky, AlmaLinux**

```
sudo dnf install -y python3 python3-pip git gh
```

**openSUSE (Leap, Tumbleweed)**

```
sudo zypper install -y python3 python3-pip git gh
```

**Any other Linux distribution**

Install Python 3.10+, Git and GitHub CLI (`gh`) with your package manager (for example `apk`, `xbps-install`, `emerge`, `nix`). Alternatively install Python with a version manager such as `uv` or `pyenv`.

**macOS**

Install Homebrew first from https://brew.sh if you do not have it, then:

```
brew install python git gh
```

**Windows 10 / 11**

Open **PowerShell** and run:

```
winget install -e --id Python.Python.3.12
winget install -e --id Git.Git
winget install -e --id GitHub.cli
```

Close PowerShell and open a new one so the new programs are found. If `winget` is missing, install Python from https://www.python.org/downloads/ (tick **"Add python.exe to PATH"** in the installer), Git from https://git-scm.com/download/win and GitHub CLI from https://cli.github.com. Use **Windows Terminal** or PowerShell. If Vietnamese characters look broken in an old console window, switch to Windows Terminal.

**Important on modern Linux:** recent Ubuntu, Debian and Arch-based systems protect the system Python. If you run `pip install ...` outside a virtual environment you get an `externally-managed-environment` error. This is expected. Always use the virtual environment from Step 4. Never use `--break-system-packages`.

You should see: version numbers for `python3` (or `py`), `git` and `gh`.

### Step 2. Log in to GitHub

```
gh auth login
```

Answer the questions like this:

1. Which account: **GitHub.com**
2. Preferred protocol: **HTTPS**
3. Authenticate Git with your GitHub credentials: **Yes**
4. How to authenticate: **Login with a web browser**

Copy the one-time code that appears, press Enter, and finish the login in the browser **with your own GitHub account**.

You should see: `Logged in as <your-username>`. You can check later with `gh auth status`.

### Step 3. Download the code

```
git clone https://github.com/wickyhien18/Vietnamese-Handwriting-OCR.git
cd Vietnamese-Handwriting-OCR
```

You should see: a folder containing `README.md`, `src/`, `scripts/` and `notebooks/`.

**Run every following command from this folder (the repository root).** Running them from a subfolder causes `ModuleNotFoundError: No module named 'src'`.

### Step 4. Run the setup script

The script creates the virtual environment, installs the dependencies, creates the local folders (`data/`, `checkpoints/`, `outputs/`) and runs the checks.

Linux and macOS (also works from fish, because the script runs in bash):

```
bash scripts/setup_local.sh
```

Windows PowerShell:

```
powershell -ExecutionPolicy Bypass -File scripts\setup_local.ps1
```

You should see: `Ran <N> tests ... OK` (every test passes; N grows as the project grows), then `OK`, and finally `Setup finished`.

The Linux/macOS script was tested on Linux. The Windows script has not been run on a real Windows machine yet, so report any error to the group.

**Activate the environment.** The script cannot activate it inside your current terminal, so do it yourself, and repeat it every time you open a new terminal:

| Shell | Command |
| ----- | ------- |
| Linux/macOS bash or zsh | `source .venv/bin/activate` |
| Linux fish (default on CachyOS) | `source .venv/bin/activate.fish` |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |
| Windows Git Bash | `source .venv/Scripts/activate` |

If PowerShell says "running scripts is disabled on this system", run this once and activate again:

```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

You should see: your prompt starts with `(.venv)`. Inside the activated environment, `python` is correct on every operating system.

<details>
<summary>Manual steps (only if the script fails)</summary>

```
python3 -m venv .venv                     # Windows: py -3 -m venv .venv
# activate it (table above), then:
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -t . -v
python scripts/validate_ocr_out.py contracts/examples/ocr_out.sample.jsonl
```

Expected: `Ran <N> tests ... OK` (no failures), then `OK`.

</details>

### Step 5. Create your own branch and push once

| Person | Role | Branch name |
| ------ | ---- | ----------- |
| Dương | A (OCR) | `a/ocr` |
| Khánh | B (corrector) | `b/corrector` |
| Hiển | C (evaluation, demo) | `c/eval` |

```
git switch -c a/ocr          # use YOUR branch name from the table
git push -u origin a/ocr     # same name here
```

You should see: your branch name on the GitHub repository page (branch dropdown). This also proves your GitHub login works.

### Step 6. Put the shared Drive folder in your My Drive

1. In Google Drive, go to **Shared with me**.
2. Right click `Vietnamese-Handwriting-OCR`, choose **Organize**, then **Add shortcut**, and pick **My Drive**.

A folder shared with you shows up under _Shared with me_, not under _My Drive_. Colab only finds it at `/content/drive/MyDrive/Vietnamese-Handwriting-OCR` after you add the shortcut. Folder shortcuts normally work in Colab, but this has not been verified for this project. If Step 7 fails on the folder, tell the group.

You should see: `Vietnamese-Handwriting-OCR` inside **My Drive**, with `notebooks`, `data`, `checkpoints` and `outputs` inside it.

### Step 7. Run the check notebook from Drive in Colab

The check notebook `00_setup_drive.ipynb` is in the `notebooks` folder of the shared Drive folder. It uses the token from Hiển to reach the private repository.

1. In Google Drive open the folder `Vietnamese-Handwriting-OCR`, then its `notebooks` folder (under **Shared with me**, or through the shortcut from Step 6). Right click `00_setup_drive.ipynb`, choose **Open with**, then **Google Colaboratory**. If Colab is not in the list, choose **Connect more apps**, search for "Colaboratory" and install it first.
2. In Colab click the **key icon (Secrets)** in the left sidebar, then **Add new secret**. Name: `GITHUB_TOKEN`. Value: the token from Hiển. Switch on **Notebook access** for this notebook. The list of secrets is shared by all your notebooks, but access is granted notebook by notebook.
3. Choose **Runtime**, then **Run all**. Allow Drive access when the pop-up appears.

Do not edit this notebook, it is shared with the whole group. Never paste the token into a cell and never commit it. The token is read-only, so Colab can download the code but cannot change it. You push code from your laptop (Step 5 and the daily workflow below).

You should see these two lines, and a folder tree that lists `notebooks`, `data`, `checkpoints` and `outputs`:

```
Drive folder is readable and writable.
GitHub access OK
```

If the first cell says the folder was not found, go back to Step 6. If the last cell fails, see the Colab rows of the troubleshooting table.

You are ready. Now read **After setup: your first session** below.

---

## After setup: your first session

The shared data is prepared once by the maintainer (see "Maintainer only: prepare the data" below). When it is ready, this is already on Drive:

```
data/uit_hwdb/<archive>.zip          the UIT-HWDB-line images and labels
data/corpus/viwiki_sentences.txt     Vietnamese sentences for role B
outputs/dataset_index.jsonl          one record per labeled image: id, image_path, gt_text
outputs/splits.json                  id -> train / val / test (the same for everybody)
outputs/dataset_report.txt           what the dataset folder looks like
```

`image_path` is always relative to the folder where the zip is extracted: **`/content/uit_hwdb`** on Colab. The start notebooks below extract it there for you.

| Role | Person | Start notebook (Drive: `notebooks/`) |
| ---- | ------ | ------------------------------------ |
| A (OCR) | Dương | `ocr_00_start.ipynb` |
| B (corrector) | Khánh | `corrector_00_start.ipynb` |
| C (evaluation) | Hiển | `00_prepare_data.ipynb` first, then your own notebooks |

How to use a start notebook:

1. In Google Drive open `Vietnamese-Handwriting-OCR/notebooks/`, right click your start notebook, choose **Open with**, then **Google Colaboratory**.
2. In Colab choose **File**, then **Save a copy in Drive**, and work in the copy. The notebook in the shared folder must stay clean for everybody.
3. In the copy, click the **key icon (Secrets)** and switch on **Notebook access** for `GITHUB_TOKEN`. Role A also switches on a GPU (**Runtime**, **Change runtime type**).
4. **Runtime**, then **Run all**. The notebook ends with a short list of what to do next.

The start notebooks in Drive are copies of the originals in the repo (`notebooks/`). When a notebook changes in the repo, the maintainer uploads the new version over the old one in Drive.

Colab cannot push to GitHub (the token is read-only). To keep your own notebook in git, choose **File**, **Download .ipynb**, put the file into `notebooks/` of your clone and push from your laptop. To test code that you pushed to your branch, set `BRANCH` in the first cell to your branch name.

Two rules for training on Colab: save a checkpoint to Drive (`checkpoints/`) after **every epoch**, because sessions get cut, and copy the images to the Colab disk instead of reading them from Drive.

---

## Maintainer only: create the Colab token (Hiển)

Teammates cannot create their own token for this repository. GitHub documents that each fine-grained token is limited to resources owned by a single user or organization, and lists "repositories where the user is an outside or repository collaborator" among the cases fine-grained tokens do not support (checked in the GitHub documentation on 6 Nov 2026). So the repository owner creates **one** read-only token and shares it privately:

1. GitHub, top right profile picture, **Settings**, **Developer settings**, **Personal access tokens**, **Fine-grained tokens**, **Generate new token**. Menu names can differ slightly.
2. Resource owner: your own account (`wickyhien18`).
3. Expiration: a date after the project deadline (for example 15 Dec 2026).
4. Repository access: **Only select repositories**, then choose `Vietnamese-Handwriting-OCR`.
5. Repository permissions: **Contents: Read-only**. Leave everything else on "No access" (Metadata read-only is added automatically).
6. **Generate token** and copy it right away: GitHub shows it only once.
7. Send it to Dương and Khánh in a **private message** each. If it ever leaks, delete it in the same settings page and create a new one.

---

## Maintainer only: prepare the data (once)

Before the first run, upload the four notebooks (`00_setup_drive`, `00_prepare_data`, `ocr_00_start`, `corrector_00_start`) from `notebooks/` of your clone into `notebooks/` of the shared Drive folder. Then open `00_prepare_data.ipynb` from Drive with **Open with**, **Google Colaboratory**, switch on notebook access for `GITHUB_TOKEN`, and **Run all**. It does these things in order:

1. Downloads UIT-HWDB-line from the Google Drive link in the dataset README into `data/uit_hwdb/`. If Google blocks the download, make a copy by hand from the link in the README and put the file into that folder.
2. Prints the layout of the dataset and saves it as `outputs/dataset_report.txt`.
3. Extracts the images on the Colab disk, then builds `outputs/dataset_index.jsonl` (`scripts/build_index.py`) and `outputs/splits.json` (`scripts/make_splits.py`).
4. Builds the sentence corpus for role B from Vietnamese Wikipedia (`scripts/prepare_corpus.py`, dataset `wikimedia/wikipedia`, config `20231101.vi`, text under CC BY-SA 3.0 and GFDL according to the dataset card).

`build_index.py` guesses the label format (JSON, CSV/TSV/TXT lines, or one text file per image). It was tested only on made-up layouts. If it stops with an error, send the printed output and `dataset_report.txt` to the group so the script can be adjusted to the real layout.

Then tell Dương and Khánh that the data is ready.

---

## Daily workflow

- Each person works on their own branch: `a/ocr`, `b/corrector`, `c/eval`.

```
git pull                         # before you start working each day
git add .
git commit -m "Describe what changed"
git push                         # first push of a new branch: git push -u origin <branch>
```

- Merge into `main` with a pull request on GitHub.
- Never commit data, model weights or secrets. They are already blocked by `.gitignore`: keep it that way.
- All code, comments and identifiers are written in English.
- Changes to `contracts/` or `src/common/` need all three people to agree.
- Notebooks are edited in Colab, then downloaded and committed from your laptop (Colab has only read access).

### First cell of every Colab notebook

```python
from google.colab import drive, userdata

drive.mount("/content/drive")

BRANCH = "main"  # use your own branch (a/ocr, b/corrector, c/eval) to test code you have pushed

token = userdata.get("GITHUB_TOKEN")
!rm -rf /content/repo
!git clone -q https://{token}@github.com/wickyhien18/Vietnamese-Handwriting-OCR.git /content/repo
!git -C /content/repo checkout -q {BRANCH}
%cd /content/repo
!pip install -q -r requirements.txt

import sys
sys.path.insert(0, "/content/repo")

DRIVE_ROOT = "/content/drive/MyDrive/Vietnamese-Handwriting-OCR"
DATA_DIR = f"{DRIVE_ROOT}/data"
CKPT_DIR = f"{DRIVE_ROOT}/checkpoints"
OUT_DIR = f"{DRIVE_ROOT}/outputs"
```

For every new notebook that uses `GITHUB_TOKEN`, open **Secrets** and switch on **Notebook access** again: the secret is shared by all your notebooks, but access is granted notebook by notebook. The clone lives in the Colab machine and disappears when the session ends. Colab only **reads** code (the token is read-only). Push code from your laptop, and save results (checkpoints, `ocr_out.jsonl`) to Drive. The token is stored inside the clone's git settings, so never share that folder and never print `git remote -v` in a notebook you share.

---

## Repository layout and who owns what

```
contracts/          data contracts shared by everyone (schema + sample file)
scripts/            validators, dataset tools (inspect_dataset.py, build_index.py, make_splits.py, prepare_corpus.py)
                    and the local setup scripts
docs/               assignments and setup checklist
src/common/         shared utilities (Unicode normalization, diacritic stripping, index reading)
src/ocr/            role A: OCR model
src/corrector/      role B: diacritic correction model
src/eval/           role C: evaluation code
src/stubs/          fake components so nobody has to wait for anybody
demo/               role C: demo app
notebooks/          Colab notebooks (originals; copies live in Drive notebooks/):
                    00_setup_drive, 00_prepare_data, ocr_00_start, corrector_00_start;
                    your own notebooks use the prefixes ocr_*, corrector_*, eval_*
tests/              unit tests
data/ checkpoints/ outputs/    local only, NEVER committed (ignored by git)
```

| Role                 | You may edit                                            | Do not edit            |
| -------------------- | ------------------------------------------------------- | ---------------------- |
| A (OCR)              | `src/ocr/`, `notebooks/ocr_*`                           | other people's folders |
| B (corrector)        | `src/corrector/`, `notebooks/corrector_*`               | other people's folders |
| C (evaluation, demo) | `src/eval/`, `demo/`, `scripts/`, `contracts/`, `docs/` | other people's folders |

`src/common/` and `contracts/` are shared. Changing them needs agreement from all three people.

## Google Drive layout

Git holds code. **Google Drive holds everything big**: datasets, model checkpoints and generated outputs.

```
Vietnamese-Handwriting-OCR/
  notebooks/          copies of the start notebooks, opened with Colab
  data/uit_hwdb/
  data/corpus/
  checkpoints/ocr/
  checkpoints/corrector/
  outputs/
```

Role A writes `ocr_out.jsonl` into `outputs/`. Role C reads it from there. `outputs/` also holds `dataset_index.jsonl`, `splits.json` and `dataset_report.txt`; `data/corpus/` holds `viwiki_sentences.txt`.

## The data contracts (frozen since 8 Nov)

Schemas, examples and details are in `contracts/README.md`. There are four contracts:

1. **`dataset_index.jsonl`** (written by Hiển): one record per labeled image with `id`, `image_path` (relative to `/content/uit_hwdb`), `gt_text`, and `split` or `writer_id` only when the dataset has them for every record. Check it with:

```
python scripts/validate_dataset_index.py --index outputs/dataset_index.jsonl
```

2. **`splits.json`** (written by Hiển): record id to `train`, `val` or `test`. Check it together with the index by adding `--splits outputs/splits.json`.

3. **`ocr_out.jsonl`** (written by Dương): one JSON object per line with the fields `id`, `split`, `image_path`, `gt_text`, `ocr_text`, and optionally `ocr_conf`. Full definition in `contracts/ocr_out.schema.json`. Check any file with:

```
python scripts/validate_ocr_out.py path/to/ocr_out.jsonl
```

4. **`correct(text)`** (written by Khánh) returns `{"text": str, "conf": [float]}`: one confidence per syllable, the same number of syllables as the input, and only diacritics change. Rules and a worked example are in `contracts/corrector_api.md`. Check your function with `check_correct_output` from `src/common/contract_checks.py`. A fake version lives in `src/stubs/identity_corrector.py`.

**All strings must be Unicode NFC.** One Vietnamese letter can be stored as one precomposed character or as a base letter plus a combining mark. If the ground truth and the OCR output use different forms, the evaluation reports fake errors. Use `normalize_nfc` from `src/common/text_utils.py`.

---

## Troubleshooting

| Problem | Fix |
| ------- | --- |
| `python3: command not found` (Linux/macOS) | Python is not installed, see Step 1. |
| `'python' is not recognized` (Windows) | Use `py` instead, or reinstall Python with "Add python.exe to PATH" ticked. |
| `gh: command not found` | Install GitHub CLI, see Step 1. |
| `git clone`: "Repository not found" | You have not accepted the invitation (Step 0), or `gh auth status` shows the wrong GitHub account (log in again, Step 2). |
| `git push` rejected, asks for a password | Log in with `gh auth login` (Step 2). GitHub does not accept account passwords. |
| `error: externally-managed-environment` | You ran `pip` outside the virtual environment. Activate `.venv` first (Step 4). |
| `The virtual environment was not created successfully ... ensurepip` (Ubuntu/Debian) | `sudo apt install python3-venv`, delete the `.venv` folder and create it again. |
| Same error on Arch-based systems | `sudo pacman -S python-pip`, delete `.venv` and create it again. |
| PowerShell: "running scripts is disabled" | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, then activate again. |
| `source: ... activate: No such file` on fish | Use `source .venv/bin/activate.fish`. |
| `ModuleNotFoundError: No module named 'src'` | Run commands from the repository root, not from a subfolder. |
| `00_prepare_data`: the download cell fails | Google sometimes blocks large downloads. Open the link from the dataset README, make a copy or download it, put the file into `data/uit_hwdb/` by hand, then run the next cells. |
| `build_index.py`: "only N of M images got a label" | The script could not read the label format. Send its full output and `outputs/dataset_report.txt` to the group so the script can be adjusted. |
| Colab check says the Drive folder was not found | Add the shortcut to My Drive (Step 6), then run the notebook again. |
| Colab: `SecretNotFoundError` | The secret does not exist. Add a secret named exactly `GITHUB_TOKEN` (Step 7). |
| Colab: `NotebookAccessError` | Switch on **Notebook access** for `GITHUB_TOKEN` in this notebook (Step 7). |
| Colab: "GitHub access FAILED" | The secret name is not exactly `GITHUB_TOKEN`, notebook access is off, the token was copied incompletely, or it expired. Ask Hiển for a new one. |
| Vietnamese letters look garbled in the terminal | Use Windows Terminal, or any UTF-8 terminal on Linux/macOS. |
| Validator says a field is "not NFC-normalized" | Run the text through `normalize_nfc` before writing the file. |
