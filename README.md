# Vietnamese Handwriting OCR + Diacritic Correction

A three-person machine learning project.

```
handwritten line image --> OCR (role A) --> diacritic corrector (role B) --> corrected text
                                   role C: evaluation, integration, demo
```

Research question: how much does a diacritic-correction step reduce OCR errors on handwritten Vietnamese, and when does it make things worse (changing a word that was already right)?

See also: [`docs/ASSIGNMENTS.md`](docs/ASSIGNMENTS.md) (who does what and when) and [`docs/SETUP_CHECKLIST.md`](docs/SETUP_CHECKLIST.md) (step-by-step setup checklist).

---

## 1. Repository layout and who owns what

```
contracts/          data contracts shared by everyone (schema + sample file)
scripts/            validators and the local setup scripts (setup_local.sh, setup_local.ps1)
docs/               assignments and setup checklist
src/common/         shared utilities (Unicode normalization, diacritic stripping)
src/ocr/            role A: OCR model
src/corrector/      role B: diacritic correction model
src/eval/           role C: evaluation code
src/stubs/          fake components so nobody has to wait for anybody
demo/               role C: demo app
notebooks/          Colab notebooks (prefix them: ocr_*, corrector_*, eval_*)
tests/              unit tests
data/ checkpoints/ outputs/    local only, NEVER committed (ignored by git)
```

| Role | You may edit | Do not edit |
|---|---|---|
| A (OCR) | `src/ocr/`, `notebooks/ocr_*` | other people's folders |
| B (corrector) | `src/corrector/`, `notebooks/corrector_*` | other people's folders |
| C (evaluation, demo) | `src/eval/`, `demo/`, `scripts/`, `contracts/`, `docs/` | other people's folders |

`src/common/` and `contracts/` are shared. Changing them needs agreement from all three people.

---

## 2. Set up your computer

You need two things: **Python 3.10 or newer** and **Git**. The project was tested on Python 3.12. Newer versions should work, older than 3.10 is untested.

Check what you already have:

```
python3 --version    # Linux / macOS
git --version
```

On Windows use `py --version` instead of `python3 --version`.

### Linux

Pick your distribution.

**Ubuntu, Debian, Linux Mint, Pop!_OS**

```
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
```

`python3-venv` is a separate package on these systems. Without it, creating a virtual environment fails.

**CachyOS, Arch Linux, Manjaro, EndeavourOS** (Arch-based, uses `pacman`)

```
sudo pacman -Syu
sudo pacman -S python git
```

On Arch-based systems the `python` package normally includes `venv`. If creating the environment later fails with an `ensurepip` error, also run `sudo pacman -S python-pip`.

CachyOS uses the **fish** shell by default. Activation commands differ for fish, see section 3.

**Fedora, RHEL, Rocky, AlmaLinux**

```
sudo dnf install -y python3 python3-pip git
```

**openSUSE (Leap, Tumbleweed)**

```
sudo zypper install -y python3 python3-pip git
```

**Any other distribution**

Install Python 3.10+ and Git with your package manager (for example `apk`, `xbps-install`, `emerge`, `nix`). Alternatively install Python with a version manager such as `uv` or `pyenv`.

**Important on modern Linux:** recent Ubuntu, Debian and Arch-based systems protect the system Python. If you run `pip install ...` outside a virtual environment you get an `externally-managed-environment` error. This is expected. Always use the virtual environment described in section 3. Never use `--break-system-packages`.

### macOS

Easiest route is Homebrew. If you do not have it, install it from https://brew.sh first.

```
brew install python git
```

Alternative: install Python from https://www.python.org/downloads/ and install Git by running `git --version` in Terminal (macOS offers to install the Xcode command line tools).

### Windows 10 / 11

Open **PowerShell** and run:

```
winget install -e --id Python.Python.3.12
winget install -e --id Git.Git
```

Close PowerShell and open a new one so the new programs are found. If `winget` is missing, install Python from https://www.python.org/downloads/ (tick **"Add python.exe to PATH"** in the installer) and Git from https://git-scm.com/download/win.

Use **Windows Terminal** or PowerShell. If Vietnamese characters look broken in an old console window, switch to Windows Terminal.

---

## 3. Get the code and run the checks

Run all commands from the **repository root** (the folder that contains `README.md`). Running them from a subfolder causes `ModuleNotFoundError: No module named 'src'`.

**Shortcut (recommended):** one script creates the virtual environment, installs the dependencies, creates the local folders (`data/`, `checkpoints/`, `outputs/`) and runs the checks. It does the same as sections 3.1 to 3.4.

Linux and macOS (also works from fish, because the script runs in bash):

```
bash scripts/setup_local.sh
```

Windows PowerShell:

```
powershell -ExecutionPolicy Bypass -File scripts\setup_local.ps1
```

It must end with `Setup finished`. The script cannot activate the environment inside your current terminal, so activate it yourself with the command for your shell (section 3.1) whenever you open a new terminal. The Linux/macOS script was tested on Linux; the Windows script has not been run on a real Windows machine yet, so report any error to the group.

The manual steps follow, in case you prefer them or the script fails.

### 3.1 Create and activate a virtual environment

**Linux and macOS (bash or zsh)**

```
python3 -m venv .venv
source .venv/bin/activate
```

**Linux (fish shell, default on CachyOS)**

```
python3 -m venv .venv
source .venv/bin/activate.fish
```

**Windows PowerShell**

```
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell says "running scripts is disabled on this system", run this once, then activate again:

```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**Windows Command Prompt (cmd.exe)**

```
py -3 -m venv .venv
.venv\Scripts\activate.bat
```

**Windows Git Bash**

```
py -3 -m venv .venv
source .venv/Scripts/activate
```

When it works, your prompt starts with `(.venv)`. Every new terminal needs the activation command again.

### 3.2 Install dependencies

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

(Inside the activated environment, `python` is correct on every operating system.)

### 3.3 Run the tests

```
python -m unittest discover -s tests -t . -v
```

Expected result: `Ran 5 tests ... OK`.

### 3.4 Validate the sample data file

```
python scripts/validate_ocr_out.py contracts/examples/ocr_out.sample.jsonl
```

Expected result: `OK`.

---

## 4. Git and GitHub workflow

### 4.1 One person, once: publish the repository

1. On github.com create a new **private** repository (empty: do not add a README, `.gitignore` or license).
2. Unzip the project, open a terminal in the folder, then:

```
git init -b main
git add .
git commit -m "Initial project skeleton"
git remote add origin https://github.com/<OWNER>/<REPO>.git
git push -u origin main
```

3. In the repository settings on GitHub, add the other two people as collaborators.

If `git push` asks for a password, GitHub does not accept account passwords. Use GitHub CLI (`gh auth login`, install from https://cli.github.com) or a personal access token, or set up an SSH key.

### 4.2 Everyone else: get the code

```
git clone https://github.com/<OWNER>/<REPO>.git
cd <REPO>
```

Then follow section 3.

### 4.3 Daily rules

- Each person works on their own branch: `a/ocr`, `b/corrector`, `c/eval`.

```
git switch -c a/ocr        # first time only (use your own branch name)
git add .
git commit -m "Describe what changed"
git push -u origin a/ocr
```

- Merge into `main` with a pull request on GitHub.
- Before you start working each day: `git pull`.
- Never commit data, model weights or secrets. They are already blocked by `.gitignore`: keep it that way.
- All code, comments and identifiers are written in English.

---

## 5. The two data contracts (frozen after 8 Nov)

1. **`ocr_out.jsonl`**: one JSON object per line with the fields `id`, `split` (`train`, `val` or `test`), `image_path`, `gt_text`, `ocr_text`, and optionally `ocr_conf`. Full definition in `contracts/ocr_out.schema.json`. Check any file with:

```
python scripts/validate_ocr_out.py path/to/ocr_out.jsonl
```

2. **`correct(text)`** returns `{"text": str, "conf": [float]}`. Syllables are separated by spaces, so `len(conf) == len(text.split())`. A fake version lives in `src/stubs/identity_corrector.py`.

**All strings must be Unicode NFC.** One Vietnamese letter can be stored as one precomposed character or as a base letter plus a combining mark. If the ground truth and the OCR output use different forms, the evaluation reports fake errors. Use `normalize_nfc` from `src/common/text_utils.py`.

---

## 6. Google Drive and Colab

Git holds code. **Google Drive holds everything big**: datasets, model checkpoints and generated outputs.

Create one shared Drive folder and share it with the other two people as **Editor**:

```
ml-project/
  data/uit_hwdb/
  data/corpus/
  checkpoints/ocr/
  checkpoints/corrector/
  outputs/
```

Role A writes `ocr_out.jsonl` into `outputs/`. Role C reads it from there.

### Create the folder automatically

The notebook `notebooks/00_setup_drive.ipynb` creates the structure above in My Drive and checks read/write access. The owner runs it once, then shares the `ml-project` folder with the others as Editor.

A folder shared with you shows up under *Shared with me*, not under *My Drive*. In the Drive web page right click it, choose *Organize*, then *Add shortcut*, and put the shortcut in *My Drive*. Then the path `/content/drive/MyDrive/ml-project` works for you as well. Folder shortcuts normally work in Colab, but this has not been verified for this project. If the path does not exist, find the real one in the Colab file browser and change `DRIVE_ROOT` and the other path variables.

### First cell of every Colab notebook

The repository is private, so Colab needs a token to clone it:

1. On GitHub create a **fine-grained personal access token** with read-only access to *Contents* for this one repository.
2. In Colab open the key icon (**Secrets**), add a secret named `GITHUB_TOKEN`, and switch on notebook access.
3. Never paste the token into a notebook cell and never commit it.

```python
from google.colab import drive, userdata

drive.mount("/content/drive")

token = userdata.get("GITHUB_TOKEN")
!git clone https://{token}@github.com/<OWNER>/<REPO>.git /content/repo
%cd /content/repo
!pip install -q -r requirements.txt

DATA_DIR = "/content/drive/MyDrive/ml-project/data"
CKPT_DIR = "/content/drive/MyDrive/ml-project/checkpoints"
OUT_DIR = "/content/drive/MyDrive/ml-project/outputs"
```

Colab runtimes reset, so the clone disappears when the session ends. Commit and push your work from Colab, or save results to Drive.

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `python3: command not found` (Linux/macOS) | Python is not installed, see section 2. |
| `'python' is not recognized` (Windows) | Use `py` instead, or reinstall Python with "Add python.exe to PATH" ticked. |
| `error: externally-managed-environment` | You ran `pip` outside the virtual environment. Activate `.venv` first (section 3.1). |
| `The virtual environment was not created successfully ... ensurepip` (Ubuntu/Debian) | `sudo apt install python3-venv`, delete the `.venv` folder and create it again. |
| Same error on Arch-based systems | `sudo pacman -S python-pip`, delete `.venv` and create it again. |
| PowerShell: "running scripts is disabled" | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, then activate again. |
| `source: ... activate: No such file` on fish | Use `source .venv/bin/activate.fish`. |
| `ModuleNotFoundError: No module named 'src'` | Run commands from the repository root, not from a subfolder. |
| `git push` rejected, asks for a password | Use `gh auth login`, a personal access token, or SSH (section 4.1). |
| Vietnamese letters look garbled in the terminal | Use Windows Terminal, or any UTF-8 terminal on Linux/macOS. |
| Validator says a field is "not NFC-normalized" | Run the text through `normalize_nfc` before writing the file. |
