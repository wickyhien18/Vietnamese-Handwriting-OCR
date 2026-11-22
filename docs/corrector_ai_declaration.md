# Khai báo sử dụng AI: vai B (mục 12)

Đưa vào phụ lục của báo cáo. Khánh cần đọc lại và đánh dấu từng dòng trước khi nộp. Vấn đáp có thể hỏi bất kỳ đoạn mã hay chứng minh nào dưới đây.

## Công cụ

- **Claude** (Anthropic), dùng qua ứng dụng Claude, các phiên từ 08/11/2026 đến 22/11/2026.
- **Gemini trong Colab**: một lần, nó tự đề xuất sửa ô kiểm tra trong `00_setup_drive.ipynb`. Bản sửa đó đã bị **hủy**, notebook được khôi phục về bản gốc.

## Claude đã làm gì

| Phần | Nội dung AI viết | Khánh đã đọc hiểu |
|---|---|---|
| `src/corrector/syllable.py` | Tách âm tiết thành chữ gốc, dấu mũ/móc, dấu thanh; quy tắc vị trí dấu thanh | [ ] |
| `src/corrector/noise.py` | Bộ sinh nhiễu (phần tự cài sẽ bị hỏi vấn đáp) | [ ] |
| `src/corrector/corpus.py` | Làm sạch, tách câu, bỏ trùng, chia tập, đo trùng lặp | [ ] |
| `src/corrector/ngram.py` | Đếm unigram và bigram, làm mịn add-one, nội suy | [ ] |
| `src/corrector/viterbi.py` | Viterbi, độ tin cậy cục bộ (phần tự cài sẽ bị hỏi vấn đáp) | [ ] |
| `src/corrector/calibration.py` | Temperature scaling, tìm kiếm tam phân, ECE (phần tự cài sẽ bị hỏi vấn đáp) | [ ] |
| `src/corrector/correctors.py`, `api.py` | Ba mức sửa, hợp đồng `correct()`, `get_corrector()` | [ ] |
| `src/corrector/tagger_data.py`, `tagger.py` | Transformer (attention tự viết), dữ liệu, huấn luyện | [ ] |
| `src/corrector/evaluate.py`, `experiments.py`, `ocr_adapt.py`, `reference_check.py` | Đo độ chính xác, 5 phần gập, thích nghi với OCR thật, đối chiếu thư viện | [ ] |
| `tests/test_corrector_*.py` | Toàn bộ test của vai B | [ ] |
| `notebooks/corrector_01` đến `corrector_06` | Toàn bộ các notebook | [ ] |
| `docs/corrector_proofs.md` | Bản nháp của ba chứng minh và ba bài tập; đáp án được kiểm bằng `tests/test_corrector_exercises.py` | [ ] |
| `docs/corrector_datasheet.md`, `corrector_model_card.md`, `corrector_reproduce.md` | Bản nháp, số liệu lấy từ các file JSON do notebook tạo | [ ] |

## Khánh đã làm gì

- Cài môi trường, chạy các notebook trên Colab, kiểm tra kết quả và gửi số liệu cho nhóm.
- Quyết định chọn kho Wikipedia, train lại transformer lâu hơn, và làm theo danh sách việc của Hiển.
- *(Khánh bổ sung các phần tự sửa hoặc tự viết, nếu có.)*

## Cam kết

Mọi số liệu trong báo cáo đều lấy từ các file JSON do notebook tạo ra. AI không tạo ra số liệu. Đáp án bài tập được kiểm bằng test tự động.
