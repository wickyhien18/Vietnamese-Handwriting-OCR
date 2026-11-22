# Tái lập kết quả của vai B (mục 11)

Mọi bước dùng seed cố định. Chạy lại trên một runtime Colab mới phải ra cùng số, trừ phần tagger: trên GPU, PyTorch có thể lệch rất nhỏ giữa các lần chạy.

## Thư viện

- `requirements.txt`: numpy. Đủ cho hai mức `frequency` và `bigram`, và cho mọi test không cần GPU.
- `requirements-corrector.txt`: torch, datasets, matplotlib, nltk, hmmlearn. Colab có sẵn tất cả trừ hmmlearn; notebook 06 tự cài hmmlearn.

## Thứ tự chạy (Colab, nhánh `b/corrector` hoặc `main` sau khi merge)

| # | Notebook | Runtime | Thời gian | Kết quả |
|---|---|---|---|---|
| 1 | `corrector_05_clean_corpus.ipynb` | CPU | ~15 phút | `data/corpus/` (kho chính), `outputs/corrector_corpus_report.json`, `outputs/corrector_corpus_removed_lines.json` |
| 2 | `corrector_06_final.ipynb` | T4 GPU | ~1,5 giờ | `checkpoints/corrector/` (ngram.pkl.gz, viterbi.json, tagger.pt, tagger_seed*.pt, default_level.json), `outputs/corrector_eval.json`, `outputs/corrector_reference_check.json`, `outputs/corrector_conf_hist.png` |
| 3 (tùy chọn) | `corrector_04_real_ocr.ipynb` | T4 GPU | ~20 phút | nhiễu theo OCR thật, `tagger_ocr.pt`, `outputs/corrector_real_ocr_eval.json` |

Notebook 06 lưu kết quả từng bước vào `outputs/corrector_stages/`. Nếu Colab ngắt, chạy lại toàn bộ: các bước đã xong được đọc lại, không tính lại. Muốn tính lại bước nào thì đặt tên bước đó vào `FORCE`.

Các notebook `corrector_01` đến `corrector_03` là lần chạy đầu trên kho bản 1. Notebook 05 và 06 làm lại toàn bộ trên kho chính.

## Một lệnh cho hai mức n-gram (không cần GPU)

```
python -m src.corrector.experiments --corpus-dir data/corpus --out outputs/corrector_eval_ngram.json
```

Lệnh này đọc 1.000.000 câu đầu của `train.txt`, chia thành 5 khối liên tục, và với mỗi khối thì huấn luyện trên 4 khối còn lại. Sau đó nó chấm 2.000 câu trải đều trong khối đó, với nhiễu seed 123. Kết quả in ra mean ± std (độ lệch chuẩn mẫu) cho `frequency` và `bigram`.

## Seed và siêu tham số

| | Giá trị |
|---|---|
| Nhiễu khi chấm | `make_pairs(..., seed=123)`, `NoiseConfig()` mặc định |
| p_same của Viterbi | đo trên 20.000 câu train đầu tiên, nhiễu seed 1 |
| n-gram | nội suy λ = 0,8; bỏ bigram gặp dưới 2 lần; k = 8 ứng viên |
| Tagger | seed 0, 1, 2, 3, 4; 300.000 câu; 2 epoch; batch 128; AdamW, lr 5e-4, weight decay 0,01; warm-up 1.000 bước; d = 256, 4 head, 4 lớp, ff 1.024, dropout 0,1 |
| Hiệu chỉnh | temperature scaling trên 5.000 câu val của kho; ECE 10 khoảng bằng nhau, trọng số theo số mẫu |

Siêu tham số được đặt **trước**, hoặc chỉ chọn trên val. `test.txt` của kho chỉ chấm một lần (bước `test_once`). 201 dòng test của UIT-HWDB không được dùng ở đây.

## Tự kiểm

1. Mở runtime Colab mới, chạy ô đầu của notebook 06 (clone, mount Drive), rồi:
   ```python
   from src.corrector.api import correct
   correct("ban")
   ```
2. `python -m unittest discover -s tests -t .`: mọi test phải qua. Trên máy không có torch, nltk hoặc hmmlearn, các test cần những thư viện đó sẽ tự bỏ qua.
