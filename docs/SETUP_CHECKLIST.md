# Setup checklist

Tick each box. Use the README for the details of each command. Commands are for the repository root.

## Part 1: one person creates the shared things (proposed: Hiển)

- [ ] Create a **private** GitHub repository (empty, no README or license).
- [ ] Unzip the project and push it (README section 4.1).
- [ ] Add Dương and Khánh as collaborators (needs their GitHub usernames).
- [ ] In Google Colab open `notebooks/00_setup_drive.ipynb` (upload it or open it from GitHub) and run cells 1 and 3. This creates `ml-project/` in your My Drive.
- [ ] In the Drive web page right click `ml-project`, Share, add the other two as **Editor**.

| Person | GitHub username | Gmail used for Drive |
|---|---|---|
| Dương | | |
| Khánh | | |
| Hiển | | |

## Part 2: everyone, on your own laptop

- [ ] Install Python 3.10+ and Git (README section 2, pick your operating system).
- [ ] Clone the repository: `git clone https://github.com/<OWNER>/<REPO>.git` then `cd <REPO>`.
- [ ] Run the setup script:
  - Linux or macOS: `bash scripts/setup_local.sh`
  - Windows PowerShell: `powershell -ExecutionPolicy Bypass -File scripts\setup_local.ps1`
- [ ] The script must end with `Setup finished`. It runs 5 tests and validates the sample file.
- [ ] Create your branch: `git switch -c a/ocr` (Dương), `git switch -c b/corrector` (Khánh), `git switch -c c/eval` (Hiển).
- [ ] Make a first tiny commit and push it (`git push -u origin <your-branch>`) to prove that your GitHub login works.

## Part 3: everyone, in Google Colab

- [ ] In Drive: right click the shared `ml-project` folder, Organize, Add shortcut, place it in My Drive.
- [ ] On GitHub create a fine-grained personal access token with **read-only** access to *Contents* for this repository only.
- [ ] In Colab open the key icon (Secrets), add a secret called `GITHUB_TOKEN` and switch on notebook access. Never paste the token into a cell.
- [ ] Open `notebooks/00_setup_drive.ipynb`, edit `<OWNER>` and `<REPO>` in cell 4, run cells 1, 3 and 4.
- [ ] Expected output: `Drive folder is readable and writable.` and `GitHub access OK`.

## Part 4: report back in the group chat

Copy, fill in and send:

```
Setup done.
OS: <Linux distro / macOS / Windows>
Tests: 5 passed
Drive: read/write OK
GitHub in Colab: OK
My branch: <branch name>
```

If any step fails, send the exact error text. Most problems are listed in the README troubleshooting table.
