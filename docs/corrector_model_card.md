# Model card: `correct()` (bộ sửa dấu tiếng Việt, vai B)

Số liệu lấy từ `outputs/corrector_eval.json` và `outputs/corrector_reference_check.json`, do `corrector_06_final.ipynb` tạo ngày 22/11/2026 trên kho chính (`docs/corrector_datasheet.md`).

## Mục đích

Nhận một dòng văn bản OCR tiếng Việt (NFC, âm tiết cách nhau bằng khoảng trắng) và trả về dòng đã sửa **dấu thanh và dấu mũ/móc/đ**, kèm một độ tin cậy cho mỗi âm tiết:

```python
from src.corrector.api import correct
correct("nhieu ban doc da bay to")   # {"text": "nhiều bạn đọc đã bày tỏ", "conf": [...]}
```

Hợp đồng nằm ở `contracts/corrector_api.md`: giữ số âm tiết, chỉ đổi dấu, giữ chữ hoa và dấu câu. Đây là bước sau OCR trong pipeline của nhóm. Vai C dùng `conf` và ngưỡng τ để quyết định có nhận gợi ý sửa hay không.

**Không dùng để:** sửa chữ cái nền sai (ví dụ "hục" thành "học"), thêm hoặc bớt âm tiết, sửa chính tả nói chung, hay sửa văn bản ngoài tiếng Việt.

## Ba mức

| Mức | Cách làm | Tự cài | Thư viện |
|---|---|---|---|
| `frequency` | Chọn dạng có dấu thường gặp nhất của âm tiết | Toàn bộ | numpy |
| `bigram` | Mô hình Markov ẩn bigram (nội suy với unigram add-one) + Viterbi | Toàn bộ | numpy |
| `tagger` | Transformer 4 lớp, d = 256, chọn một ứng viên cho mỗi âm tiết | Attention, khối, đầu chấm điểm, dữ liệu, lịch học | PyTorch: Embedding, Linear, LayerNorm, cross-entropy, AdamW |

Mức mặc định được chọn trên 636 dòng OCR val của UIT-HWDB (CER thấp nhất khi nhận mọi gợi ý), ghi trong `checkpoints/corrector/default_level.json`: **`tagger`** (CER 13,72 %, so với 15,00 % của `bigram` và 17,85 % của `frequency`). Lưu ý: cả ba mức đều làm CER tăng so với không sửa (12,27 %); xem phần chỉ số.

## Dữ liệu

- **Huấn luyện:** kho Wikipedia tiếng Việt đã làm sạch (`docs/corrector_datasheet.md`). Cặp huấn luyện tạo bằng bộ sinh nhiễu: xóa dấu cả câu (p = 0,3), xóa dấu một âm tiết (0,10), đổi dấu thanh (0,10), đổi dấu mũ/móc (0,05). Bộ sinh nhiễu **không bao giờ đổi chữ cái nền**, vì `correct()` không được phép sửa loại lỗi đó.
- **Hiệu chỉnh:** temperature scaling, khớp trên val của kho (`calibration.py`).
- **Chọn mức mặc định:** UIT-HWDB val. 201 dòng test của UIT-HWDB không được dùng cho bất kỳ lựa chọn nào.

## Chỉ số

**Trên kho văn bản** (nhiễu tổng hợp; đây **không phải** kết quả của đề):

| Mức | Độ chính xác âm tiết | Âm tiết nhập nhằng | Làm hỏng âm tiết vốn đúng |
|---|---|---|---|
| `frequency` (5 phần gập) | 72,78 % ± 0,39 | 71,50 % ± 0,27 | 26,84 % ± 0,61 |
| `bigram` (5 phần gập) | **94,74 % ± 0,21** | **94,59 % ± 0,21** | 4,44 % ± 0,25 |
| `tagger` (5 seed) | 92,04 % ± 0,08 | 91,81 % ± 0,09 | **2,30 % ± 0,06** |

Chấm một lần trên `test.txt` của kho (5.000 câu, 97.041 âm tiết): `frequency` 73,17 %, `bigram` 95,15 %, `tagger` 91,52 %, khớp với các phần gập.

Ghi chú: 5 lần của `frequency` và `bigram` là 5 phần gập theo khối liên tục (1.000.000 câu). 5 lần của `tagger` là 5 seed (0–4), mỗi seed 300.000 câu, 2 epoch, khoảng 6,3 phút một lần trên T4; mô hình đóng gói là seed 1, tốt nhất trên val. Lần chạy đầu (8/11) train tagger lâu hơn (600.000 câu, 3 epoch) và đạt 95,3 %. Tagger trong bảng thấp hơn bigram chủ yếu vì ngân sách huấn luyện nhỏ hơn.

Lần chạy đầu (kho bản 1, một lần chạy, nhiễu trộn): không sửa 59,8 %, `frequency` 73,7 %, `bigram` 95,8 %, `tagger` 95,3 %.

"Âm tiết nhập nhằng" là âm tiết mà khi bỏ dấu thì ứng với từ hai dạng có dấu trở lên trong kho huấn luyện. Độ lệch chuẩn là độ lệch chuẩn mẫu (chia cho n − 1).

**Trên OCR val thật** (636 dòng, 9.888 âm tiết được căn với nhãn, nhận **mọi** gợi ý, τ = 0):

| Mức | CER | WER | Sửa đúng | Làm hỏng | Ròng | Tỉ lệ làm hỏng | Sửa được trong số lỗi chỉ sai dấu (364) |
|---|---|---|---|---|---|---|---|
| Không sửa | 12,27 % | 34,30 % | 0 | 0 | 0 | 0 % | 0 % |
| `frequency` | 17,85 % | 49,82 % | 167 | 1.734 | −1.567 | 25,83 % | 45,9 % |
| `bigram` | 15,00 % | 38,07 % | 249 | 636 | −387 | 9,48 % | 68,4 % |
| `tagger` | **13,72 %** | **34,99 %** | 149 | **225** | **−76** | **3,35 %** | 40,9 % |

Khi nhận mọi gợi ý, **cả ba mức đều làm CER tăng**: số âm tiết bị làm hỏng nhiều hơn số được sửa. Bigram sửa được nhiều lỗi dấu nhất (68 %), nhưng cũng làm hỏng nhiều nhất. Tagger ít hại nhất. Hàm `correct()` vi phạm hợp đồng 0 lần trên cả 636 dòng ở mọi mức.

Kết quả chính thức của đề (bảng sửa đúng / sửa hỏng, quét τ, bootstrap) do Hiển đo bằng `eval_00_start.ipynb`. Với ngưỡng τ, có thể chỉ giữ lại những gợi ý thật chắc; nếu không ngưỡng nào có ích, "không sửa" thắng.

**Hiệu chỉnh (ECE, 10 khoảng bằng nhau, trọng số theo số mẫu):**

| Mức | T | ECE val kho: trước → sau | ECE OCR val: trước → sau |
|---|---|---|---|
| `frequency` | 0,989 | 0,0050 → 0,0047 | 0,229 → 0,232 |
| `bigram` | 1,184 | 0,0089 → 0,0039 | 0,262 → 0,242 |
| `tagger` | 1,045 | 0,0056 → 0,0023 | 0,261 → 0,257 |

Trên kho văn bản, độ tin cậy đã khớp rất tốt. Trên OCR thật thì không: ECE khoảng 0,23 đến 0,26. Lý do chính là phần lớn âm tiết sai của OCR sai ở chữ cái nền. Bộ sửa không sửa được những âm tiết này nhưng vẫn rất tự tin về dấu của chúng, vì độ tin cậy chỉ đo "chọn đúng dấu trong các ứng viên", không đo "âm tiết đúng". Temperature scaling khớp trên nhiễu tổng hợp nên không sửa được độ lệch này. Tỉ lệ âm tiết có conf ≥ 0,95 trên OCR val: `frequency` 29,2 %, `bigram` 60,8 %, `tagger` 75,1 % (biểu đồ `outputs/corrector_conf_hist.png`).

**Đối chiếu với thư viện** (`outputs/corrector_reference_check.json`): n-gram và nội suy khớp `nltk.lm` tuyệt đối (sai lệch lớn nhất 0 trên 13,5 triệu cặp); Viterbi trùng vét cạn ở 1.203/1.203 câu ngắn và trùng `hmmlearn` ở 50/50 mô hình ngẫu nhiên (lệch log-xác suất ≤ 3,6·10⁻¹⁵); attention tự viết khớp `torch.nn.MultiheadAttention` (lệch ≤ 1,2·10⁻⁷, sai số dấu phẩy động).

**Kích thước và thời gian nạp:** `ngram.pkl.gz` 6,6 MB; `tagger.pt` 147,6 MB. Nạp và gọi lần đầu: `frequency` 1,5 s, `bigram` 2,0 s, `tagger` 0,8 s trên Colab.

## Giới hạn (bắt buộc đọc)

1. **Trần lợi ích nhỏ.** Lỗi dấu chỉ chiếm khoảng 11 % lỗi OCR. Lỗi chữ cái nền chiếm 82,3 % và nằm ngoài phạm vi của bộ sửa. Một bộ sửa dấu hoàn hảo cũng chỉ giảm được khoảng 0,85 đến 1,38 điểm CER, nên rất có thể "không sửa" thắng ở bước quét τ. Nhóm đã chấp nhận đây là một kết quả âm tính hợp lệ.
2. **Lệch miền.** Huấn luyện trên Wikipedia, nhưng dữ liệu UIT-HWDB là văn báo chí và có câu hội thoại. Ví dụ đã thấy: "Toi mua ban ghe moi" thành "Tối mùa bán ghế mới". Độ tin cậy của những âm tiết sai này thấp (0,49 và 0,54), nên ngưỡng τ có thể chặn chúng.
3. **Độ tin cậy quá cao trên OCR thật.** ECE khoảng 0,25 trên OCR val, vì conf không tính đến lỗi chữ cái nền (xem trên). Nên chọn τ trên OCR val, không tin vào giá trị conf tuyệt đối.
4. **Nhiễu tổng hợp khác nhiễu OCR.** OCR thật hiếm khi xóa dấu cả dòng, còn bộ sinh nhiễu làm vậy với 30 % câu. `corrector_04_real_ocr.ipynb` đo xác suất nhiễu thật trên val và tinh chỉnh tagger theo đó.
5. **Chỉ chọn trong ứng viên.** Mỗi âm tiết có tối đa 8 dạng có dấu thường gặp nhất trong kho. Âm tiết lạ được giữ nguyên với độ tin cậy 1,0.
6. **Kiểu bỏ dấu.** Đầu ra luôn theo kiểu cũ ("hòa", "thủy").
7. Mô hình bigram sau khi cắt tỉa (bỏ cặp chỉ gặp một lần) không còn là phân phối xác suất chuẩn, vì tổng nhỏ hơn 1 (xem `docs/corrector_proofs.md`, Mệnh đề 2b). Điều này không ảnh hưởng đến Viterbi.

## Tái lập

Xem `docs/corrector_reproduce.md`.
