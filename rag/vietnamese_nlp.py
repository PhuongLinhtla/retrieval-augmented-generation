from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


_WORD_PATTERN = re.compile(r"[\w]+", re.UNICODE)
_YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")
_GRADE_PATTERN = re.compile(r"\b(lop|grade)\s*([1-9]|1[0-2])\b", re.IGNORECASE)
_CHAPTER_PATTERN = re.compile(
    r"\b(chuong|chapter|phan|muc|section)\s*([0-9]{1,2}[a-z]?)\b",
    re.IGNORECASE,
)

_ABBREVIATIONS = {
    "dh": "dai hoc",
    "thpt": "trung hoc pho thong",
    "thcs": "trung hoc co so",
    "sv": "sinh vien",
    "gv": "giao vien",
    "bai tap": "bai tap",
    "pt": "phuong trinh",
    "bpt": "bat phuong trinh",
    "bđt": "bat dang thuc",
}

_SENTENCE_PROTECTED = {
    "ts.": "ts<dot>",
    "ths.": "ths<dot>",
    "p.gs.": "pgs<dot>",
    "tp.hcm": "tphcm",
    "vd.": "vd<dot>",
    "v.d.": "vd<dot>",
    "v.v.": "vv<dot>",
}

_STOPWORDS = {
    "la",
    "va",
    "hoac",
    "voi",
    "cua",
    "cho",
    "trong",
    "neu",
    "thi",
    "tai",
    "duoc",
    "khong",
    "mot",
    "nhung",
    "nhieu",
    "cac",
    "nhu",
    "ve",
    "theo",
    "tu",
    "den",
    "de",
    "hay",
    "roi",
}


@dataclass(frozen=True, slots=True)
class TextUnit:
    text: str
    kind: str


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    title: str
    subject: str
    grade: str
    chapter: str
    published_year: str
    source: str


def strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_whitespace(text: str) -> str:
    normalized = text.replace("\u00a0", " ")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def normalize_text_for_matching(text: str, keep_diacritics: bool = True) -> str:
    cleaned = normalize_whitespace(text).casefold()
    if not keep_diacritics:
        cleaned = strip_diacritics(cleaned)
    cleaned = re.sub(r"[\s\u200b]+", " ", cleaned)
    return cleaned.strip()


def fix_common_ocr_errors(text: str) -> str:
    if not text:
        return ""

    cleaned = text
    # Common OCR mistakes around numeric contexts.
    cleaned = re.sub(r"(?<=\d)[lI](?=\d)", "1", cleaned)
    cleaned = re.sub(r"(?<=\d)[oO](?=\d)", "0", cleaned)
    cleaned = re.sub(r"(?<=\b)rn(?=\w)", "m", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;:!?])(\S)", r"\1 \2", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return normalize_whitespace(cleaned)


def expand_common_abbreviations(text: str) -> str:
    if not text:
        return ""

    out = f" {text} "
    for short, expanded in _ABBREVIATIONS.items():
        pattern = re.compile(rf"(?<!\w){re.escape(short)}(?!\w)", re.IGNORECASE)
        out = pattern.sub(expanded, out)
    return normalize_whitespace(out)


def tokenize_vietnamese(
    text: str,
    *,
    remove_stopwords: bool = False,
    keep_diacritics: bool = True,
) -> list[str]:
    normalized = normalize_text_for_matching(text, keep_diacritics=keep_diacritics)
    tokens = [token for token in _WORD_PATTERN.findall(normalized) if len(token) > 1]
    if not remove_stopwords:
        return tokens
    return [token for token in tokens if token not in _STOPWORDS]


def split_sentences_vietnamese(text: str) -> list[str]:
    raw = normalize_whitespace(text)
    if not raw:
        return []

    protected = raw
    for original, replacement in _SENTENCE_PROTECTED.items():
        protected = protected.replace(original, replacement)

    pieces = re.split(r"(?<=[.!?])\s+", protected)
    sentences: list[str] = []
    for piece in pieces:
        item = piece.strip()
        if not item:
            continue
        restored = item
        for original, replacement in _SENTENCE_PROTECTED.items():
            restored = restored.replace(replacement, original)
        sentences.append(restored)

    if not sentences:
        return [raw]
    return sentences


def split_text_units(text: str) -> list[TextUnit]:
    lines = [line.rstrip() for line in text.splitlines()]
    units: list[TextUnit] = []
    paragraph_buffer: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_buffer:
            return
        paragraph_text = normalize_whitespace(" ".join(paragraph_buffer))
        if paragraph_text:
            units.append(TextUnit(text=paragraph_text, kind="paragraph"))
        paragraph_buffer.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue

        is_heading = bool(
            re.match(r"^(#{1,6}\s+|chuong\s+\d+|chapter\s+\d+|\d+(\.\d+){0,3}\s+)", stripped, re.IGNORECASE)
        )
        is_bullet = bool(re.match(r"^([-*+]|\d+[.)])\s+", stripped))
        is_formula_like = bool(re.search(r"[=<>^]{1,}|\$.*\$", stripped))

        if is_heading:
            flush_paragraph()
            units.append(TextUnit(text=normalize_whitespace(stripped), kind="heading"))
            continue

        if is_bullet:
            flush_paragraph()
            units.append(TextUnit(text=normalize_whitespace(stripped), kind="bullet"))
            continue

        if is_formula_like and len(stripped) < 180:
            flush_paragraph()
            units.append(TextUnit(text=normalize_whitespace(stripped), kind="formula"))
            continue

        paragraph_buffer.append(stripped)

    flush_paragraph()
    return units


def estimate_token_count(text: str) -> int:
    if not text:
        return 0

    words = _WORD_PATTERN.findall(text)
    punctuation = re.findall(r"[.,;:!?()\[\]{}]", text)
    token_count = len(words) + max(1, len(punctuation) // 2)
    if token_count <= 0:
        token_count = max(1, len(text) // 5)
    return token_count


def classify_question_intent(question: str) -> str:
    normalized = normalize_text_for_matching(question, keep_diacritics=False)

    if re.search(r"\b(dinh nghia|la gi|khai niem)\b", normalized):
        return "definition"
    if re.search(r"\b(chung minh|vi sao|giai thich)\b", normalized):
        return "proof"
    if re.search(r"\b(cach lam|quy trinh|cac buoc|huong dan)\b", normalized):
        return "procedure"
    if re.search(r"\b(so sanh|khac nhau|giong nhau|doi chieu)\b", normalized):
        return "comparison"
    if re.search(r"\b(vi du|bai tap|minh hoa)\b", normalized):
        return "example"
    return "general"


def extract_keywords(text: str) -> list[str]:
    normalized = normalize_text_for_matching(text, keep_diacritics=False)
    labels = {
        "dinh nghia": ["dinh nghia", "khai niem"],
        "dinh ly": ["dinh ly", "bo de", "he qua"],
        "chung minh": ["chung minh", "proof"],
        "vi du": ["vi du", "minh hoa"],
        "cau hoi": ["cau hoi", "bai tap"],
        "loi giai": ["loi giai", "dap an"],
    }

    found: list[str] = []
    for label, variants in labels.items():
        if any(variant in normalized for variant in variants):
            found.append(label)
    return found


def build_contextual_header(
    *,
    subject: str,
    grade: str,
    chapter: str,
    section_title: str,
    page_number: int | None,
) -> str:
    parts: list[str] = []
    if subject.strip():
        parts.append(subject.strip())
    if grade.strip():
        parts.append(grade.strip())
    if chapter.strip():
        parts.append(chapter.strip())
    if section_title.strip():
        parts.append(section_title.strip())
    if page_number is not None and page_number > 0:
        parts.append(f"Trang {page_number}")

    if not parts:
        return ""
    return " · ".join(parts)


def extract_document_metadata(source_name: str, text: str = "") -> DocumentMetadata:
    normalized_name = source_name.strip()
    simplified = normalize_text_for_matching(normalized_name, keep_diacritics=False)
    expanded = expand_common_abbreviations(simplified)

    title = normalized_name
    if text.strip():
        first_line = normalize_whitespace(text.split("\n", 1)[0])
        if 4 <= len(first_line) <= 180:
            title = first_line

    year_match = _YEAR_PATTERN.search(expanded)
    grade_match = _GRADE_PATTERN.search(expanded)
    chapter_match = _CHAPTER_PATTERN.search(expanded)

    subject = ""
    subject_hints = {
        "toan": ["toan", "dai so", "hinh hoc"],
        "vat ly": ["vat ly", "ly"],
        "hoa hoc": ["hoa", "hoa hoc"],
        "sinh hoc": ["sinh", "sinh hoc"],
        "ngu van": ["van", "ngu van"],
        "lich su": ["lich su", "su hoc"],
        "dia ly": ["dia ly", "dia"],
        "tin hoc": ["tin hoc", "lap trinh"],
        "tieng anh": ["tieng anh", "english"],
    }

    for candidate, hints in subject_hints.items():
        if any(hint in expanded for hint in hints):
            subject = candidate
            break

    grade = ""
    if grade_match:
        grade = f"Lop {grade_match.group(2)}"

    chapter = ""
    if chapter_match:
        chapter = f"Chuong {chapter_match.group(2)}"

    published_year = year_match.group(1) if year_match else ""

    return DocumentMetadata(
        title=title.strip(),
        subject=subject,
        grade=grade,
        chapter=chapter,
        published_year=published_year,
        source=normalized_name,
    )


def expand_query_variants(query: str) -> list[str]:
    base = normalize_whitespace(query)
    if not base:
        return []

    normalized = normalize_text_for_matching(base, keep_diacritics=True)
    accentless = normalize_text_for_matching(base, keep_diacritics=False)
    expanded = expand_common_abbreviations(accentless)

    variants = [base]
    for candidate in (normalized, accentless, expanded):
        if candidate and candidate not in variants:
            variants.append(candidate)

    focused = " ".join(tokenize_vietnamese(expanded, remove_stopwords=True, keep_diacritics=False))
    if focused and focused not in variants:
        variants.append(focused)

    return variants[:4]
