"""Lab 1 starter: versioned bilingual preprocessing for Bayan."""
import re
import unicodedata

PREPROC_VERSION = "1.2.0"

TATWEEL = "ـ"

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")
_WHITESPACE_RE = re.compile(r"\s+")

_PHONE_RE = re.compile(r"(?<!\d)(?:\+?966\d{9}|05\d{8})(?!\d)")
_NATIONAL_ID_RE = re.compile(r"(?<!\d)[12]\d{9}(?!\d)")


def normalize(text: str) -> str:
    """Return deterministic Bayan normalisation while preserving task signal."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(TATWEEL, "")
    text = _HTML_TAG_RE.sub(" ", text)
    text = _REPEATED_CHAR_RE.sub(r"\1\1", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def mask_pii(text: str) -> str:
    """Mask supported phone numbers and Saudi national-ID-shaped values."""
    text = _PHONE_RE.sub("<PHONE>", text)
    text = _NATIONAL_ID_RE.sub("<NATIONAL_ID>", text)
    return text


def preprocess(text: str) -> str:
    """Apply the shared train/eval/serve preprocessing contract."""
    return mask_pii(normalize(text))
