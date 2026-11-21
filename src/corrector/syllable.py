"""Break a Vietnamese syllable into letters, shape marks and tone marks, and put it back together.

Every Vietnamese letter is seen as three parts:
    base letter  : a, e, o, u, d, ...
    shape mark   : circumflex (â ê ô), breve (ă), horn (ơ ư), or the stroke of đ
    tone mark    : grave (huyền), acute (sắc), hook (hỏi), tilde (ngã), dot below (nặng)

Example: "ộ" = base "o" + shape circumflex + tone dot below.
The noise generator changes these parts one at a time.
"""

import unicodedata

# The five tone marks as Unicode combining characters.
GRAVE = "̀"      # huyền: à
ACUTE = "́"      # sắc:   á
HOOK = "̉"       # hỏi:   ả
TILDE = "̃"      # ngã:   ã
DOT_BELOW = "̣"  # nặng:  ạ
TONES = [GRAVE, ACUTE, HOOK, TILDE, DOT_BELOW]

# The shape marks. The stroke of "đ" is not a combining character in Unicode,
# so it gets its own marker string.
CIRCUMFLEX = "̂"  # â ê ô
BREVE = "̆"       # ă
HORN = "̛"        # ơ ư
STROKE = "stroke"      # đ
SHAPES = [CIRCUMFLEX, BREVE, HORN]

# Which shape marks each base letter can carry in Vietnamese (None = no mark).
SHAPE_OPTIONS = {
    "a": [None, CIRCUMFLEX, BREVE],
    "e": [None, CIRCUMFLEX],
    "o": [None, CIRCUMFLEX, HORN],
    "u": [None, HORN],
    "d": [None, STROKE],
}

VOWELS = set("aeiouy")


def split_char(ch):
    """Return [base, shape, tone] for one character. shape and tone may be None."""
    if ch == "đ":  # đ
        return ["d", STROKE, None]
    if ch == "Đ":  # Đ
        return ["D", STROKE, None]
    parts = unicodedata.normalize("NFD", ch)
    base, shape, tone = parts[0], None, None
    for mark in parts[1:]:
        if mark in SHAPES:
            shape = mark
        elif mark in TONES:
            tone = mark
        else:
            # A mark that Vietnamese does not use: keep the character untouched.
            return [ch, None, None]
    return [base, shape, tone]


def join_char(base, shape, tone):
    """Inverse of split_char."""
    if shape == STROKE:
        return "đ" if base == "d" else "Đ"
    text = base + (shape or "") + (tone or "")
    return unicodedata.normalize("NFC", text)


def decompose(syllable):
    """"người" -> [["n",None,None], ["g",None,None], ["u",HORN,None], ["o",HORN,GRAVE], ["i",None,None]]"""
    return [split_char(ch) for ch in unicodedata.normalize("NFC", syllable)]


def compose(chars):
    """Inverse of decompose."""
    return "".join(join_char(base, shape, tone) for base, shape, tone in chars)


def tone_index(chars):
    """Index of the letter that carries the tone mark, or None if the syllable has no tone."""
    for i, (_, _, tone) in enumerate(chars):
        if tone is not None:
            return i
    return None


def tone_position(chars):
    """Index of the letter where the tone mark should be written (traditional style: "hòa", "thủy").

    Rules:
    1. Find the vowel group. In "qu" and "gi", the u / i belong to the first consonant
       when another vowel follows ("quý", "giá"), but not in "gì".
    2. If a vowel in the group has a shape mark, the tone goes on the last such vowel ("người").
    3. If the group has one vowel, the tone goes on it.
    4. If a consonant follows the group (closed syllable), the tone goes on the last vowel ("hoàn").
    5. Otherwise (open syllable), the tone goes on the second-to-last vowel ("hòa", "ngoài").
    """
    bases = [base.lower() for base, _, _ in chars]
    vowel_idx = [i for i, b in enumerate(bases) if b in VOWELS]
    if not vowel_idx:
        return None

    # Rule 1: skip the u of "qu" and the i of "gi" when another vowel follows.
    if len(vowel_idx) > 1 and vowel_idx[0] == 1:
        if (bases[0] == "q" and bases[1] == "u") or (bases[0] == "g" and bases[1] == "i"):
            vowel_idx = vowel_idx[1:]

    # Keep only the first block of consecutive vowels.
    group = [vowel_idx[0]]
    for i in vowel_idx[1:]:
        if i != group[-1] + 1:
            break
        group.append(i)

    shaped = [i for i in group if chars[i][1] is not None]
    if shaped:
        return shaped[-1]                 # rule 2
    if len(group) == 1:
        return group[0]                   # rule 3
    if group[-1] < len(chars) - 1:
        return group[-1]                  # rule 4
    return group[-2]                      # rule 5


def normalize_tone_placement(syllable):
    """Move the tone mark to the standard position: "hoà" -> "hòa", "thuỷ" -> "thủy".

    The corpus mixes both writing styles. Without this, "hoà" and "hòa" would count as
    two different words.
    """
    chars = decompose(syllable)
    current = tone_index(chars)
    if current is None:
        return compose(chars)
    target = tone_position(chars)
    if target is None or target == current:
        return compose(chars)
    tone = chars[current][2]
    chars[current][2] = None
    chars[target][2] = tone
    return compose(chars)


def split_token(token):
    """Separate punctuation around a word: '"Hà,' -> ('"', 'Hà', ',').

    The core runs from the first letter to the last letter of the token.
    """
    letters = [i for i, ch in enumerate(token) if ch.isalpha()]
    if not letters:
        return token, "", ""
    start, end = letters[0], letters[-1] + 1
    return token[:start], token[start:end], token[end:]
