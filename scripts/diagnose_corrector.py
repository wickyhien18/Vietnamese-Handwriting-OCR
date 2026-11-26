"""Chẩn đoán số liệu bộ sửa dấu (vai B) trước khi chốt báo cáo.

Chạy từ THƯ MỤC GỐC của repo (nơi có thư mục src/), trên Colab sau khi clone repo và mount Drive:

    python diagnose_corrector.py --ocr outputs/ocr_out.jsonl --ckpt checkpoints/corrector --outputs outputs

Tham số:
  --ocr      tệp OCR của vai A (cột split, gt_text, ocr_text). Dùng đúng tệp đã dùng cho Bảng 7-10.
  --ckpt     thư mục chứa tagger.pt, tagger_seed*.pt, ngram.pkl.gz, viterbi.json
  --outputs  thư mục chứa corrector_eval.json, corrector_reference_check.json (nếu có)
  --level    mức sửa dấu cần chẩn đoán (mặc định: tagger)
Chỉ ĐỌC dữ liệu và in kết quả; không ghi đè tệp nào của dự án. Gửi lại toàn bộ phần in ra.
"""
import argparse
import glob
import hashlib
import json
import os

from src.corrector import api
from src.corrector.ocr_adapt import aligned_syllables, evaluate_on_ocr, ocr_lines
from src.eval.align import corpus_error_rate


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def changed_breakdown(corrector, lines, tau):
    """Đếm token bị đổi ở ngưỡng tau theo ĐÚNG quy tắc của evaluate_on_ocr (conf >= tau thì nhận)."""
    n_tokens = changed = changed_aligned = changed_extra = sat_changed = 0
    for gt_text, ocr_text in lines:
        result = corrector.correct(ocr_text)
        old_tokens = ocr_text.split()
        aligned_positions = {pos for _, _, pos in aligned_syllables(gt_text, ocr_text)}
        for i, (old, new, c) in enumerate(zip(old_tokens, result["text"].split(), result["conf"])):
            n_tokens += 1
            if c >= tau and new != old:
                changed += 1
                if i in aligned_positions:
                    changed_aligned += 1
                else:
                    changed_extra += 1  # token OCR thừa, không có cặp trong nhãn: Bảng 9 không đếm
                if c >= 1.0:
                    sat_changed += 1
    return {"tokens": n_tokens, "changed_all_tokens": changed, "changed_in_aligned_pairs": changed_aligned,
            "changed_in_unaligned_extra_tokens": changed_extra, "changed_with_conf_exactly_1.0": sat_changed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--level", default="tagger")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.ocr, encoding="utf-8") if l.strip()]
    lines = ocr_lines(rows, "val")
    print(f"=== 0. Dữ liệu: {len(rows)} dòng trong tệp, {len(lines)} dòng val")

    # ---- A. NFC / khoảng trắng: CER 'không sửa' tính thô và sau chuẩn hóa
    raw_pairs = [(r["gt_text"], r["ocr_text"]) for r in rows if r["split"] == "val"]
    print("\n=== A. CER 'không sửa' (OCR thô) trên val")
    print(f"  thô (không NFC, không gộp khoảng trắng): {corpus_error_rate(raw_pairs):.6f}")
    print(f"  sau ocr_lines (NFC + gộp khoảng trắng) : {corpus_error_rate(lines):.6f}")
    print("  Nếu hai số khác nhau: pipeline chuẩn hóa làm đổi mẫu số/phân số của mọi hàng Bảng 8.")

    corrector = api.build_corrector(a.level, a.ckpt)

    # ---- B. Quét tau, gồm tau > 1 (không thể nhận gợi ý nào)
    print(f"\n=== B. Mức '{a.level}': quét tau")
    hdr = f"{'tau':>6} {'CER':>9} {'WER':>9} {'net':>5} | {'all_chg':>7} {'aligned':>7} {'extra':>6} {'conf=1.0':>8}"
    print(hdr)
    for tau in (0.0, 0.8, 0.95, 1.0, 1.01):
        r = evaluate_on_ocr(corrector, lines, tau=tau)
        b = changed_breakdown(corrector, lines, tau)
        print(f"{tau:>6} {r['cer_after']:>9.6f} {r['wer_after']:>9.6f} {r['net_fixed']:>5} | "
              f"{b['changed_all_tokens']:>7} {b['changed_in_aligned_pairs']:>7} "
              f"{b['changed_in_unaligned_extra_tokens']:>6} {b['changed_with_conf_exactly_1.0']:>8}")
        if tau == 1.01:
            print(f"  kiểm tra: CER(tau=1.01) == CER không sửa ? {abs(r['cer_after'] - r['cer_before']) < 1e-12}"
                  f"  (cer_before={r['cer_before']:.6f})")
    print("  Đọc: 'net' = sửa đúng - sửa hỏng, KHÔNG phải số token bị đổi. So cột all_chg ở tau=1.0 với 'net'.")
    print("  Cột 'extra' ở tau=0 so với 1.404/1.432: nếu extra = 28 thì đó là nguồn chênh lệch.")

    # ---- C. tagger.pt so với tagger_seed*.pt
    print("\n=== C. Tệp trọng số tagger")
    if a.ckpt:
        files = sorted(glob.glob(os.path.join(a.ckpt, "tagger*.pt")))
        try:
            import torch
            loaded = {os.path.basename(p): torch.load(p, map_location="cpu", weights_only=False) for p in files}
        except Exception as e:  # noqa
            loaded = {}
            print("  (không đọc được bằng torch:", repr(e), ")")
        base = loaded.get("tagger.pt")
        for p in files:
            name = os.path.basename(p)
            line = f"  {name:<18} sha256[:16]={sha(p)}"
            if name in loaded:
                line += f"  temperature={loaded[name]['temperature']}"
                if base is not None and name != "tagger.pt":
                    same = all((loaded[name]['state_dict'][k] == base['state_dict'][k]).all().item()
                               for k in base['state_dict'])
                    line += f"  trọng số giống tagger.pt: {same}"
            print(line)

    # ---- D. ECE, T, 5 hạt giống, đối chiếu thư viện (đọc từ JSON đã có)
    print("\n=== D. Số liệu đã lưu trong outputs/")
    ev = os.path.join(a.outputs, "corrector_eval.json")
    if os.path.exists(ev):
        d = json.load(open(ev, encoding="utf-8"))
        print("  default_level:", d.get("default_level"))
        for lvl, v in d["levels"].items():
            print(f"  [{lvl}] T={v.get('temperature')}  ece_ocr_val={v.get('ece_ocr_val')}")
            print(f"        ece_corpus_val={v.get('ece_corpus_val')}")
            print(f"        n_runs={v.get('n_runs')} all={v.get('all')} std={v.get('std')} chosen_seed={v.get('chosen_seed')}")
            print(f"        ocr_val_tau0={v.get('ocr_val_tau0')}")
    else:
        print("  thiếu", ev)
    rc = os.path.join(a.outputs, "corrector_reference_check.json")
    if os.path.exists(rc):
        print("  corrector_reference_check.json:")
        print(json.dumps(json.load(open(rc, encoding="utf-8")), ensure_ascii=False, indent=1)[:4000])
    else:
        print("  thiếu", rc)


if __name__ == "__main__":
    main()
