# Setup checklist

Tick each box. The numbered steps are explained in detail in the README ("Start here: full setup, in order"). Run every command from the repository root.

## Maintainer (Hiển), before the others start

- [ ] Invite Dương and Khánh as collaborators on `wickyhien18/Vietnamese-Handwriting-OCR`.
- [ ] Share the Drive folder `Vietnamese-Handwriting-OCR` with both as **Editor**.
- [ ] Create the read-only token (README, "Maintainer only: create the Colab token").
- [ ] Send the token to each of them in a private message.
- [ ] Merge the dataset tools and the start notebooks into `main`.
- [ ] Upload the four notebooks from `notebooks/` of your clone into the Drive folder `Vietnamese-Handwriting-OCR/notebooks/` (move `00_setup_drive.ipynb` there too).
- [ ] Run `notebooks/00_prepare_data.ipynb` in Colab (README, "Maintainer only: prepare the data"). Check that `outputs/` holds `dataset_index.jsonl`, `splits.json`, `dataset_report.txt`, and `data/corpus/` holds `viwiki_sentences.txt`.
- [ ] Tell Dương and Khánh that the data is ready.

| Person | GitHub username | Google account used for Drive | Invited | Drive shared | Token sent |
|---|---|---|---|---|---|
| Dương | | | [ ] | [ ] | [ ] |
| Khánh | | | [ ] | [ ] | [ ] |

## Everyone

- [ ] Step 0: accepted the GitHub invitation, see the Drive folder under *Shared with me*, have the token.
- [ ] Step 1: Python 3.10+, Git and GitHub CLI installed.
- [ ] Step 2: `gh auth login` done, `gh auth status` shows your own account.
- [ ] Step 3: repository cloned, `cd Vietnamese-Handwriting-OCR`.
- [ ] Step 4: setup script ends with `Setup finished`; environment activated, prompt starts with `(.venv)`.
- [ ] Step 5: own branch created and pushed (Dương `a/ocr`, Khánh `b/corrector`, Hiển `c/eval`).
- [ ] Step 6: Drive shortcut added to My Drive.
- [ ] Step 7: `00_setup_drive.ipynb` opened from Drive `notebooks/` in Colab, secret `GITHUB_TOKEN` saved with notebook access on, Run all printed `Drive folder is readable and writable.` and `GitHub access OK`.

## First session (after the data is ready)

- [ ] (Optional) `git pull` on your laptop to see new code.
- [ ] Dương: open `ocr_00_start.ipynb` from Drive `notebooks/` (Open with Colaboratory), File > Save a copy in Drive, switch on a GPU and notebook access in the copy, Run all.
- [ ] Khánh: open `corrector_00_start.ipynb` from Drive `notebooks/` (Open with Colaboratory), File > Save a copy in Drive, switch on notebook access in the copy, Run all.
- [ ] Post in the group chat what you saw (counts, the three samples, or the corpus numbers).

If any step fails, send the exact error text. Most problems are in the README troubleshooting table.
