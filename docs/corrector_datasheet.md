# Datasheet: kho văn bản tiếng Việt của vai B

Kho này dùng để huấn luyện và đánh giá bộ sửa dấu (`src/corrector/`). Ô nào ghi **[điền sau khi chạy]** thì lấy số từ file được chỉ ra, sau khi chạy `notebooks/corrector_05_clean_corpus.ipynb`.

## 1. Mục đích

Tạo cặp huấn luyện (câu bị làm nhiễu dấu, câu đúng) cho bộ sửa dấu, bằng cách làm hỏng dấu của câu sạch (`src/corrector/noise.py`). Kho không chứa ảnh hay nhãn của UIT-HWDB.

## 2. Nguồn và giấy phép

| | |
|---|---|
| Nguồn | Wikipedia tiếng Việt qua Hugging Face: dataset `wikimedia/wikipedia`, cấu hình `20231101.vi` (bản xuất ngày 01/11/2023, khoảng 1,29 triệu bài) |
| Giấy phép | CC BY-SA 3.0 và GFDL, theo thẻ dữ liệu của dataset (như README của nhóm đã ghi) |
| Cách tải | Đọc dạng stream, theo thứ tự có sẵn trong dataset, dừng khi đủ `MAX_SENTENCES = 1.500.000` câu |
| Ngày tải | Bản 1 (`data/corpus_v1/`): 08/11/2026. Bản chính (`data/corpus/`): 22/11/2026 (`built_at` = 2026-11-22 08:53:31 giờ UTC của Colab) |

## 3. Tiền xử lý (`src/corrector/corpus.py`)

Với mỗi bài:
1. Chuẩn hóa Unicode NFC, bỏ phần trong ngoặc `(...)` và `[...]` (phần lớn là tên nước ngoài, năm, chú thích), gộp khoảng trắng.
2. Tách câu sau `. ! ? …` khi ký tự kế tiếp là chữ hoa, chữ số, dấu ngoặc kép hoặc ngoặc mở. **Không** cắt sau chữ viết tắt một chữ cái ("Ulysses S. Grant", "A.G.").
3. Giữ câu dài **4 đến 40 âm tiết**, có ít nhất 90 % từ chỉ gồm chữ cái tiếng Việt, và ít nhất 30 % từ có dấu (loại câu tiếng Anh và câu không dấu).
4. Đưa dấu thanh về vị trí kiểu cũ: "hoà" → "hòa", "thuỷ" → "thủy" (`normalize_tone_placement`).

Cho cả kho:

5. **Bỏ trùng** sau khi chuẩn hóa giống `scripts/check_text_overlap.py` (chữ thường, bỏ mọi ký tự không phải chữ, số, dấu). Ví dụ "Hà Nội." và "hà nội ," được coi là một câu.
6. **Chia train / val / test = 98 / 1 / 1 %** theo băm MD5 của câu đã chuẩn hóa. Hai câu gần trùng luôn rơi vào cùng một tập, nên số câu trùng nguyên văn giữa các tập bằng 0 ngay từ khi dựng kho.

## 4. Các phép loại trùng (mục 6 của danh sách việc)

| | train ∩ val (trùng nguyên văn) | train ∩ test | val có chung 8-gram với train | test có chung 8-gram với train |
|---|---|---|---|---|
| Bản 1, Hiển đo | 39 câu | 44 câu | 523 / 13.461 | 543 / 13.405 |
| Bản 1, `corpus_overlap` (mọi câu) | 119 / 15.057 | 125 / 15.061 | 2.340 / 15.057 | 2.367 / 15.061 |
| Bản chính | **0** / 15.122 (kiểm tra bằng `assert`) | **0** / 15.028 | 2.201 / 15.122 | 2.176 / 15.028 |

Có **334 dòng** của `corpus_v1/train.txt` trùng với val hoặc test. Danh sách các dòng đó nằm trong `outputs/corrector_corpus_removed_lines.json`. Toàn bộ số liệu nằm trong `outputs/corrector_corpus_report.json`.

Số của Hiển (39 và 44) và số của `corpus_overlap` (119 và 125) dùng cùng cách chuẩn hóa nhưng khác phạm vi: `corpus_overlap` đếm trên mọi câu của val và test, kể cả câu ngắn dưới 8 từ; số của Hiển tính trên 13.461 và 13.405 câu. Với cả hai cách đếm, bản chính đều bằng 0.

Bản 1 có câu trùng vì cách chia cũ băm câu **nguyên văn**: hai câu chỉ khác dấu câu hoặc chữ hoa bị tách sang hai tập. Bản chính băm câu **đã chuẩn hóa**, nên không còn trường hợp này. Trùng 8-gram không bằng 0 vì các cụm từ thường gặp lặp lại tự nhiên giữa các câu khác nhau; nhóm không đặt mục tiêu bằng 0 cho số này.

So với dòng val và test của UIT-HWDB (`check_text_overlap.py --corpus`): bản 1 không trùng nguyên dòng, mức "major" là 1,6 % (val) và 3,6 % (test), đều dưới ngưỡng 10 %, nên giữ nguyên. Bản chính: **[điền sau khi chạy]** (in ra ở bước 4 của notebook 05).

## 5. Kích thước

| | Bản 1 (08/11) | Bản chính |
|---|---|---|
| Số bài đã đọc | 39.957 | 54.860 |
| Câu train / val / test | 1.469.986 / 15.057 / 15.061 | 1.469.903 / 15.122 / 15.028 |
| Câu trùng đã bỏ | 121.428 | 159.875 (bỏ trùng sau khi chuẩn hóa nên nhiều hơn) |
| Âm tiết trong train | 32.538.245 | 29.617.815 (câu tối đa 40 thay vì 50 âm tiết) |

## 6. Hạn chế đã biết

- **Lệch chủ đề:** kho lấy các bài **đầu tiên** theo thứ tự của dataset (54.860 trên khoảng 1,29 triệu bài), không lấy ngẫu nhiên, nên một số chủ đề bị dư (ví dụ nhiều câu về Thụy Điển).
- **Lệch miền:** văn Wikipedia mang tính bách khoa. Nhãn UIT-HWDB là văn báo chí và có câu hội thoại. Câu ngôi thứ nhất như "tôi mua bàn ghế" rất hiếm trong kho.
- Còn sót một ít tên riêng nước ngoài trong câu, vì mỗi câu được phép có tới 10 % từ không phải tiếng Việt.
- **Kiểu bỏ dấu:** kho dùng kiểu cũ ("hòa"). Nếu nhãn UIT-HWDB dùng kiểu mới ("hoà"), bộ sửa sẽ bị tính sai ở những âm tiết đó. `corrector_04_real_ocr.ipynb` đo tỉ lệ này trên val.
- Kho không chứa thông tin cá nhân ngoài tên người công khai trên Wikipedia.
