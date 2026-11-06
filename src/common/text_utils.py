import unicodedata


def normalize_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def strip_diacritics(text: str) -> str:
    text = normalize_nfc(text)
    text = text.replace("\u0111", "d").replace("\u0110", "D")
    decomposed = unicodedata.normalize("NFD", text)
    kept = [ch for ch in decomposed if not unicodedata.combining(ch)]
    return unicodedata.normalize("NFC", "".join(kept))


def split_syllables(text: str) -> list:
    return text.split()
