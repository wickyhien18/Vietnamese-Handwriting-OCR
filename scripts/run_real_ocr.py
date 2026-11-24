"""Run the frozen OCR model once on the hand-collected real handwriting test set.

The vocabulary is rebuilt from the train split of splits.json exactly as in the training
notebook, the checkpoint is loaded, and every image goes through the same preprocessing
as the validation run (grayscale, height 32, scaled to [-1, 1], batches of 32 padded with
white). The only addition is the EXIF rotation fix (about 20 of the 90 photos are stored
sideways); --no-exif-fix reproduces the first run. Greedy and beam decoders are the
hand-written ones. Existing output files are never overwritten unless --force is given.

Usage (Colab with Drive mounted, from the repository root):
  python scripts/run_real_ocr.py --tag run1
Output: <real-dir>/real_ocr_out_greedy_<tag>.jsonl and real_ocr_out_beam_<tag>.jsonl
"""

import argparse
import json
import os
import unicodedata

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

DEFAULT_ROOT = "/content/drive/MyDrive/Vietnamese-Handwriting-OCR"


def nfc(text):
    return unicodedata.normalize("NFC", text)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_vocab(index_path, splits_path, expected_train_lines):
    index = read_jsonl(index_path)
    with open(splits_path, "r", encoding="utf-8") as handle:
        splits = json.load(handle)
    train = [rec for rec in index if splits.get(rec["id"]) == "train"]
    if expected_train_lines and len(train) != expected_train_lines:
        raise SystemExit("wrong splits.json: %d train lines, expected %d" % (len(train), expected_train_lines))
    vocab = sorted(list({ch for rec in train for ch in nfc(rec["gt_text"])}))
    char2idx = {char: idx + 1 for idx, char in enumerate(vocab)}
    char2idx["<blank>"] = 0
    idx2char = {idx: char for char, idx in char2idx.items()}
    return char2idx, idx2char


def preprocess_line_image(img_path, fixed_height=32, fix_orientation=True):
    with Image.open(img_path) as img:
        if fix_orientation:
            img = ImageOps.exif_transpose(img)
        gray = img.convert("L")
        w, h = gray.size
        new_w = max(16, int(round(w * (fixed_height / float(h)))))
        resized = gray.resize((new_w, fixed_height), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32)
    return (arr / 255.0 - 0.5) / 0.5


class LineDataset(Dataset):
    def __init__(self, records, img_dir, char2idx, fix_orientation):
        self.records = records
        self.img_dir = img_dir
        self.char2idx = char2idx
        self.fix_orientation = fix_orientation

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        arr = preprocess_line_image(os.path.join(self.img_dir, rec["image_path"]), 32, self.fix_orientation)
        label = [self.char2idx[c] for c in nfc(rec["gt_text"]) if c in self.char2idx]
        return torch.from_numpy(arr).unsqueeze(0), torch.tensor(label, dtype=torch.long), arr.shape[1]


def collate_fn(batch):
    images, labels, widths = zip(*batch)
    max_w = max(widths)
    padded_images = torch.ones((len(images), 1, 32, max_w), dtype=torch.float32)
    for i, img in enumerate(images):
        padded_images[i, :, :, :img.shape[2]] = img
    input_lengths = torch.tensor([(w // 4) + 1 for w in widths], dtype=torch.long)
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    return padded_images, padded_labels, input_lengths, target_lengths


class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0), nn.BatchNorm2d(512), nn.ReLU(True)
        )
        self.rnn = nn.LSTM(512, hidden_size, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x).squeeze(2).permute(0, 2, 1)
        rnn_out, _ = self.rnn(conv)
        output = F.log_softmax(self.fc(self.dropout(rnn_out)), dim=2)
        return output.permute(1, 0, 2)


def ctc_greedy_decode(log_probs, idx2char, blank=0):
    preds = np.argmax(log_probs, axis=1)
    chars, prev = [], None
    for p in preds:
        if p != blank and p != prev:
            chars.append(idx2char[int(p)])
        prev = int(p)
    return "".join(chars)


def _log_add(a, b):
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    if a > b:
        return a + np.log1p(np.exp(b - a))
    return b + np.log1p(np.exp(a - b))


def ctc_beam_search_decode(log_probs, idx2char, beam_width=10, top_k=15, blank=0):
    T, C = log_probs.shape
    if T == 0:
        return ""
    k = min(top_k, C)
    beams = {(): (0.0, -np.inf)}
    for t in range(T):
        lp = log_probs[t]
        top_chars = np.argsort(lp)[-k:]
        new_beams = {}
        for prefix, (p_b, p_nb) in beams.items():
            p_total = _log_add(p_b, p_nb)
            nb_b, nb_nb = new_beams.get(prefix, (-np.inf, -np.inf))
            nb_b = _log_add(nb_b, p_total + float(lp[blank]))
            new_beams[prefix] = (nb_b, nb_nb)
            for c in top_chars:
                c = int(c)
                if c == blank:
                    continue
                c_lp = float(lp[c])
                if len(prefix) > 0 and c == prefix[-1]:
                    same_b, same_nb = new_beams.get(prefix, (-np.inf, -np.inf))
                    same_nb = _log_add(same_nb, p_nb + c_lp)
                    new_beams[prefix] = (same_b, same_nb)
                    ext_prefix = prefix + (c,)
                    ext_b, ext_nb = new_beams.get(ext_prefix, (-np.inf, -np.inf))
                    ext_nb = _log_add(ext_nb, p_b + c_lp)
                    new_beams[ext_prefix] = (ext_b, ext_nb)
                else:
                    ext_prefix = prefix + (c,)
                    ext_b, ext_nb = new_beams.get(ext_prefix, (-np.inf, -np.inf))
                    ext_nb = _log_add(ext_nb, p_total + c_lp)
                    new_beams[ext_prefix] = (ext_b, ext_nb)
        sorted_beams = sorted(new_beams.items(), key=lambda item: _log_add(item[1][0], item[1][1]), reverse=True)[:beam_width]
        beams = dict(sorted_beams)
    best_prefix, _ = max(beams.items(), key=lambda item: _log_add(item[1][0], item[1][1]))
    return "".join(idx2char[i] for i in best_prefix)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drive-root", default=DEFAULT_ROOT)
    parser.add_argument("--real-dir", default=None, help="default: <drive-root>/data/handwriting-test")
    parser.add_argument("--index-dir", default=None, help="folder with dataset_index.jsonl and splits.json; default: <drive-root>/outputs")
    parser.add_argument("--ckpt", default=None, help="default: <drive-root>/checkpoints/ocr/best_crnn.pth")
    parser.add_argument("--tag", default="", help="suffix of the output file names, for example run1")
    parser.add_argument("--no-exif-fix", action="store_true", help="skip the EXIF rotation fix (reproduces the first run)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--expected-lines", type=int, default=90)
    parser.add_argument("--expected-train-lines", type=int, default=5369)
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    args = parser.parse_args()

    real_dir = args.real_dir or os.path.join(args.drive_root, "data", "handwriting-test")
    index_dir = args.index_dir or os.path.join(args.drive_root, "outputs")
    ckpt = args.ckpt or os.path.join(args.drive_root, "checkpoints", "ocr", "best_crnn.pth")
    suffix = "_" + args.tag if args.tag else ""
    out_greedy = os.path.join(real_dir, "real_ocr_out_greedy%s.jsonl" % suffix)
    out_beam = os.path.join(real_dir, "real_ocr_out_beam%s.jsonl" % suffix)
    if not args.force:
        for path in (out_greedy, out_beam):
            if os.path.exists(path):
                raise SystemExit("%s already exists; use another --tag or --force" % path)

    char2idx, idx2char = build_vocab(os.path.join(index_dir, "dataset_index.jsonl"),
                                     os.path.join(index_dir, "splits.json"), args.expected_train_lines)
    real = read_jsonl(os.path.join(real_dir, "real_dataset_index.jsonl"))
    if args.expected_lines and len(real) != args.expected_lines:
        raise SystemExit("expected %d lines, found %d" % (args.expected_lines, len(real)))
    unseen = sorted({ch for rec in real for ch in nfc(rec["gt_text"]) if ch not in char2idx})
    print("label characters missing from the training vocabulary:", unseen)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(len(char2idx)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    dataset = LineDataset(real, real_dir, char2idx, fix_orientation=not args.no_exif_fix)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    greedy_rows, beam_rows = [], []
    pointer = 0
    with torch.no_grad():
        for images, _, input_lengths, _ in loader:
            outputs = model(images.to(device)).permute(1, 0, 2).cpu().numpy()
            for b in range(outputs.shape[0]):
                valid_t = min(int(input_lengths[b]), outputs.shape[1])
                log_probs = outputs[b, :valid_t]
                rec = real[pointer]
                pointer += 1
                base = {"id": rec["id"], "split": "test", "image_path": rec["image_path"], "gt_text": nfc(rec["gt_text"])}
                greedy_rows.append({**base, "ocr_text": nfc(ctc_greedy_decode(log_probs, idx2char))})
                beam_rows.append({**base, "ocr_text": nfc(ctc_beam_search_decode(log_probs, idx2char, beam_width=args.beam_width))})

    write_jsonl(out_greedy, greedy_rows)
    write_jsonl(out_beam, beam_rows)
    print("wrote %d lines to %s and %s (EXIF fix: %s)" % (len(beam_rows), out_greedy, out_beam, not args.no_exif_fix))
    for row in beam_rows[:5]:
        print("GT :", row["gt_text"])
        print("OCR:", row["ocr_text"])


if __name__ == "__main__":
    main()
