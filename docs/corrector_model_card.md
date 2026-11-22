# Model card: `correct()` (bộ sửa dấu tiếng Việt, vai B)

Số liệu cuối cùng lấy từ `outputs/corrector_eval.json` (do `corrector_06_final.ipynb` tạo). Ô nào ghi **[điền sau khi chạy]** thì chép số từ file đó.

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

Mức mặc định được chọn trên 636 dòng OCR val của UIT-HWDB (CER thấp nhất khi nhận mọi gợi ý), ghi trong `checkpoints/corrector/default_level.json`: **[điền sau khi chạy]**.

## Dữ liệu

- **Huấn luyện:** kho Wikipedia tiếng Việt đã làm sạch (`docs/corrector_datasheet.md`). Cặp huấn luyện tạo bằng bộ sinh nhiễu: xóa dấu cả câu (p = 0,3), xóa dấu một âm tiết (0,10), đổi dấu thanh (0,10), đổi dấu mũ/móc (0,05). Bộ sinh nhiễu **không bao giờ đổi chữ cái nền**, vì `correct()` không được phép sửa loại lỗi đó.
- **Hiệu chỉnh:** temperature scaling, khớp trên val của kho (`calibration.py`).
- **Chọn mức mặc định:** UIT-HWDB val. 201 dòng test của UIT-HWDB không được dùng cho bất kỳ lựa chọn nào.

## Chỉ số

**Trên kho văn bản** (nhiễu tổng hợp; đây **không phải** kết quả của đề):

| Mức | Độ chính xác âm tiết | Âm tiết nhập nhằng | Làm hỏng âm tiết vốn đúng |
|---|---|---|---|
| `frequency` (5 phần gập) | **[điền sau khi chạy]** mean ± std | **[điền sau khi chạy]** | **[điền sau khi chạy]** |
| `bigram` (5 phần gập) | **[điền sau khi chạy]** | **[điền sau khi chạy]** | **[điền sau khi chạy]** |
| `tagger` (5 seed) | **[điền sau khi chạy]** | **[điền sau khi chạy]** | **[điền sau khi chạy]** |

Lần chạy đầu (kho bản 1, một lần chạy, nhiễu trộn): không sửa 59,8 %, `frequency` 73,7 %, `bigram` 95,8 %, `tagger` 95,3 %.

"Âm tiết nhập nhằng" là âm tiết mà khi bỏ dấu thì ứng với từ hai dạng có dấu trở lên trong kho huấn luyện. Độ lệch chuẩn là độ lệch chuẩn mẫu (chia cho n − 1).

**Trên OCR val thật** (do `corrector_06_final.ipynb` đo, nhận mọi gợi ý): CER trước và sau, tỉ lệ làm hỏng, ECE trước và sau hiệu chỉnh. **[điền sau khi chạy]** Kết quả chính thức của đề (bảng sửa đúng / sửa hỏng, quét τ, bootstrap) do Hiển đo bằng `eval_00_start.ipynb`.

## Giới hạn (bắt buộc đọc)

1. **Trần lợi ích nhỏ.** Lỗi dấu chỉ chiếm khoảng 11 % lỗi OCR. Lỗi chữ cái nền chiếm 82,3 % và nằm ngoài phạm vi của bộ sửa. Một bộ sửa dấu hoàn hảo cũng chỉ giảm được khoảng 0,85 đến 1,38 điểm CER, nên rất có thể "không sửa" thắng ở bước quét τ. Nhóm đã chấp nhận đây là một kết quả âm tính hợp lệ.
2. **Lệch miền.** Huấn luyện trên Wikipedia, nhưng dữ liệu UIT-HWDB là văn báo chí và có câu hội thoại. Ví dụ đã thấy: "Toi mua ban ghe moi" thành "Tối mùa bán ghế mới". Độ tin cậy của những âm tiết sai này thấp (0,49 và 0,54), nên ngưỡng τ có thể chặn chúng.
3. **Nhiễu tổng hợp khác nhiễu OCR.** OCR thật hiếm khi xóa dấu cả dòng, còn bộ sinh nhiễu làm vậy với 30 % câu. `corrector_04_real_ocr.ipynb` đo xác suất nhiễu thật trên val và tinh chỉnh tagger theo đó.
4. **Chỉ chọn trong ứng viên.** Mỗi âm tiết có tối đa 8 dạng có dấu thường gặp nhất trong kho. Âm tiết lạ được giữ nguyên với độ tin cậy 1,0.
5. **Kiểu bỏ dấu.** Đầu ra luôn theo kiểu cũ ("hòa", "thủy").
6. Mô hình bigram sau khi cắt tỉa (bỏ cặp chỉ gặp một lần) không còn là phân phối xác suất chuẩn, vì tổng nhỏ hơn 1 (xem `docs/corrector_proofs.md`, Mệnh đề 2b). Điều này không ảnh hưởng đến Viterbi.

## Tái lập

Xem `docs/corrector_reproduce.md`.
