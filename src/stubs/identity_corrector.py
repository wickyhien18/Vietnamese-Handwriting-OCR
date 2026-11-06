def correct(text: str) -> dict:
    syllables = text.split()
    return {"text": text, "conf": [1.0] * len(syllables)}
