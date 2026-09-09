"""Lab 4 starter: per-model Arabic normalisation profiles."""
import re
import unicodedata
from dataclasses import dataclass

TATWEEL = "ـ"
ALEF_VARIANTS = "آأإٱ"  # آ أ إ ٱ
ALEF_MAKSURA = "ى"  # ى
TEH_MARBUTA = "ة"  # ة
HAMZA_ON_WAW = "ؤ"  # ؤ
HAMZA_ON_YEH = "ئ"  # ئ

_DIACRITICS_RE = re.compile(
    "[" + "".join(
        [
            "ً",  # tanwin fath
            "ٌ",  # tanwin damm
            "ٍ",  # tanwin kasr
            "َ",  # fatha
            "ُ",  # damma
            "ِ",  # kasra
            "ّ",  # shadda
            "ْ",  # sukun
            "ٓ",  # maddah above
            "ٰ",  # superscript alef
        ]
    ) + "]"
)
_ALEF_RE = re.compile(f"[{ALEF_VARIANTS}]")


@dataclass(frozen=True)
class ArabicProfile:
    name: str
    dediacritize: bool = False


def normalize_arabic(text: str, profile: ArabicProfile) -> str:
    """Apply the Bayan Arabic normalisation contract.

    Always: Unicode NFKC, tatweel removal, hamza-on-alef -> bare alef,
    alef maksura -> yeh, teh marbuta -> heh, hamza-on-waw/yeh -> bare waw/yeh.
    Diacritics are stripped only when profile.dediacritize is True, so a
    "display" profile can keep them for showing to users while a "model"
    profile (dediacritize=True) gets the fully normalised form fed to tokenizers.
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace(TATWEEL, "")
    normalized = _ALEF_RE.sub("ا", normalized)
    normalized = normalized.replace(ALEF_MAKSURA, "ي")
    normalized = normalized.replace(TEH_MARBUTA, "ه")
    normalized = normalized.replace(HAMZA_ON_WAW, "و")
    normalized = normalized.replace(HAMZA_ON_YEH, "ي")

    if profile.dediacritize:
        normalized = _DIACRITICS_RE.sub("", normalized)

    return normalized


_disambiguator = None


def _get_disambiguator():
    global _disambiguator
    if _disambiguator is None:
        from camel_tools.disambig.mle import MLEDisambiguator

        _disambiguator = MLEDisambiguator.pretrained()
    return _disambiguator


def segment(text: str) -> list[str]:
    """Clitic-segment Arabic text using CAMeL Tools' D3 scheme (MLE disambiguator).

    D3 marks clitic boundaries with "+" (proclitic/enclitic) and word boundaries
    with "_", e.g. "مرجعه" -> "مَرْجِع_+هُ" -> ["مرجع", "ه"]. Falls back to the raw
    word when the disambiguator has no analysis for it.
    """
    from camel_tools.tokenizers.word import simple_word_tokenize
    from camel_tools.utils.dediac import dediac_ar

    words = simple_word_tokenize(text)
    if not words:
        return []

    disambig = _get_disambiguator().disambiguate(words)

    segments = []
    for word, result in zip(words, disambig):
        d3seg = result.analyses[0].analysis.get("d3seg", word) if result.analyses else word
        if d3seg == "NOAN":  # CAMeL Tools' no-analysis marker
            d3seg = word
        for piece in d3seg.replace("+", "_").split("_"):
            piece = dediac_ar(piece).strip()
            if piece:
                segments.append(piece)
    return segments
