"""Levenshtein alignment with backtrace (hand-written, no external library).

Convention used everywhere in this project:
  ref = ground truth, hyp = OCR output (the thing being scored).
  Rows of the table follow prefixes of hyp, columns follow prefixes of ref.
  table[i][j] = minimum number of edits to turn hyp[:i] into ref[:j].

The functions work on any two sequences (strings for characters, lists for
syllables or words), as long as elements can be compared with ==.
"""

MATCH = "match"
SUB = "sub"   # hyp has a different element than ref at this position
INS = "ins"   # hyp has an extra element that ref does not have
DEL = "del"   # hyp is missing an element that ref has


def edit_distance_table(ref, hyp):
    rows = len(hyp) + 1
    cols = len(ref) + 1
    table = [[0] * cols for _ in range(rows)]
    for j in range(cols):
        table[0][j] = j
    for i in range(rows):
        table[i][0] = i
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if hyp[i - 1] == ref[j - 1] else 1
            table[i][j] = min(
                table[i - 1][j] + 1,         # extra element in hyp
                table[i][j - 1] + 1,         # element missing from hyp
                table[i - 1][j - 1] + cost,  # match or substitution
            )
    return table


def edit_distance(ref, hyp):
    return edit_distance_table(ref, hyp)[len(hyp)][len(ref)]


def align(ref, hyp):
    """Return a list of (kind, ref_element, hyp_element), from left to right.

    Backtrace starts at the bottom-right cell. At every step it prefers the
    diagonal move (match or substitution), then an extra hyp element, then a
    missing hyp element. The fixed preference makes the result deterministic.
    """
    table = edit_distance_table(ref, hyp)
    i, j = len(hyp), len(ref)
    ops = []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if hyp[i - 1] == ref[j - 1] else 1
            if table[i][j] == table[i - 1][j - 1] + cost:
                kind = MATCH if cost == 0 else SUB
                ops.append((kind, ref[j - 1], hyp[i - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and table[i][j] == table[i - 1][j] + 1:
            ops.append((INS, None, hyp[i - 1]))
            i -= 1
        else:
            ops.append((DEL, ref[j - 1], None))
            j -= 1
    ops.reverse()
    return ops


def count_ops(ops):
    counts = {MATCH: 0, SUB: 0, INS: 0, DEL: 0}
    for kind, _, _ in ops:
        counts[kind] += 1
    return counts


def error_rate(ref, hyp):
    """Edit distance divided by len(ref). CER for strings, WER for word lists."""
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else float("inf")
    return edit_distance(ref, hyp) / len(ref)


def corpus_error_rate(pairs):
    """Total edits divided by total reference length over many (ref, hyp) pairs."""
    total_edits = 0
    total_ref = 0
    for ref, hyp in pairs:
        total_edits += edit_distance(ref, hyp)
        total_ref += len(ref)
    return total_edits / total_ref if total_ref else 0.0
