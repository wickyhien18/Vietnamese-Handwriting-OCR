"""Điền nốt kết quả Phần C: C1(c), C8, C10(a).

Cần:  pip install editdistance
Chạy từ bất kỳ thư mục nào bên trong kho mã:  python3 fill_partC.py
In ra các dòng để dán vào báo cáo (thay cho các chỗ "[điền kết quả chạy của bạn]").

verify_partBC.py vẫn chỉ dùng numpy và thư viện chuẩn. Tệp này cần thêm src/eval/align.py của kho
và thư viện editdistance, nên nhớ ghi cả hai tệp ở Phụ lục C của báo cáo.
"""
import itertools
import os
import random
import sys


def find_repo_root():
    for start in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        here = start
        while True:
            if os.path.exists(os.path.join(here, "src", "eval", "align.py")):
                return here
            parent = os.path.dirname(here)
            if parent == here:
                break
            here = parent
    raise SystemExit("Không tìm thấy src/eval/align.py: hãy chạy tệp này từ bên trong kho mã.")


sys.path.insert(0, find_repo_root())
import editdistance  # noqa: E402
from src.eval.align import SUB, align, edit_distance  # noqa: E402


def rand_text(chars, low, high):
    size = random.randint(low, high)
    return "".join(random.choice(chars) for _ in range(size))


# ---------------------------------------------------------------- C1(c)
# edit_distance của align.py phải trùng editdistance, và chi phí căn chỉnh phải bằng d.
random.seed(7)
alphabet = "abcdeoun \u00e1\u1ea1\u00f4\u0111"
mismatch = 0
for _ in range(2000):
    a, b = rand_text(alphabet, 0, 12), rand_text(alphabet, 0, 12)
    d = edit_distance(a, b)
    cost = sum(1 for kind, _x, _y in align(b, a) if kind != "match")
    if d != editdistance.eval(a, b) or cost != d:
        mismatch += 1
print("C1(c): 2000 cặp, %d cặp lệch" % mismatch)

# ---------------------------------------------------------------- C8
# Bốn tính chất của Mệnh đề 8 trên các bộ ba chuỗi ngẫu nhiên; mọi số phải bằng 0.
random.seed(11)
alphabet = "abo \u00e1\u1ea1"
bad = {"identity": 0, "symmetry": 0, "triangle": 0, "lower": 0, "upper": 0}
for _ in range(5000):
    a, b, c = rand_text(alphabet, 0, 8), rand_text(alphabet, 0, 8), rand_text(alphabet, 0, 8)
    dab, dba = edit_distance(a, b), edit_distance(b, a)
    dbc, dac = edit_distance(b, c), edit_distance(a, c)
    bad["identity"] += (dab == 0) != (a == b)
    bad["symmetry"] += dab != dba
    bad["triangle"] += dac > dab + dbc
    bad["lower"] += dab < abs(len(a) - len(b))
    bad["upper"] += dab > max(len(a), len(b))
print("C8: 5000 bộ ba, số vi phạm:", bad)

# ---------------------------------------------------------------- C10(a)
# Vét cạn mọi h' trong V(h): min d(h', r) phải bằng d(sigma h, sigma r) (Mệnh đề 18), và
# số đếm theo nhãn S_dau không bao giờ vượt trần chính xác E (Mệnh đề 19(b)).
variants = {"a": ["a", "\u00e1", "\u1ea1"],
            "o": ["o", "\u00f4", "\u1ed1"],
            "d": ["d", "\u0111"], "n": ["n"]}
chars = [c for vs in variants.values() for c in vs]


def strip(text):
    return "".join(next(b for b, vs in variants.items() if ch in vs) for ch in text)


random.seed(1)
violations = strictly_below = equal = exceeds = 0
for _ in range(3000):
    ref, hyp = rand_text(chars, 1, 5), rand_text(chars, 1, 5)
    options = [variants[strip(c)] for c in hyp]
    best = min(edit_distance(ref, "".join(p)) for p in itertools.product(*options))
    claim = edit_distance(strip(ref), strip(hyp))
    violations += best != claim
    s_dia = sum(1 for kind, r, h in align(ref, hyp) if kind == SUB and strip(r) == strip(h))
    exact_reduction = edit_distance(ref, hyp) - claim
    strictly_below += s_dia < exact_reduction
    equal += s_dia == exact_reduction
    exceeds += s_dia > exact_reduction
print("C10(a): 3000 cặp, %d cặp vi phạm Mệnh đề 18" % violations)
print("C10(a): số đếm theo nhãn = trần chính xác ở %d cặp, nhỏ hơn ở %d cặp (%.1f%%), lớn hơn ở %d cặp" % (
    equal, strictly_below, 100.0 * strictly_below / 3000, exceeds))
