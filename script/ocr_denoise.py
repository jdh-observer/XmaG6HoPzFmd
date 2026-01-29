# ocr_denoise.py
from pathlib import Path
import re
import unicodedata
from typing import Optional, Tuple, List
from wordfreq import zipf_frequency

__all__ = [
    "ZIPF_THRESHOLD",
    "MIN_COMBINED_LEN",
    "SPLIT_MIN_LEN",
    "MIN_PART_LEN",
    "NO_SPLIT_RIGHT_SUFFIXES",
    "NO_SPLIT_LEFT_PREFIXES",
    "normalize_spaces",
    "preprocess_text",
    "preprocess_file_line_by_line",
    "merge_split_words_in_line",
    "stitch_hyphenated_lines",
    "stitch_spacebroken_lines",
    "split_runons_in_line",
]

### All parameters that can be additionally tuned to cleanup the text ###

ZIPF_THRESHOLD = 3.5
MIN_COMBINED_LEN = 4

# Tighter run-on splitting
SPLIT_MIN_LEN = 12
MIN_PART_LEN = 4

# Never split into these right-hand whole-word suffix fragments
NO_SPLIT_RIGHT_SUFFIXES = {
    "ing","ions","ment","ments","ness","ship","hood","ism","ist","ists",
    "able","ible","less","ward","wards","wise","tion","tions","sion","sions",
    "ally","ality","ities","ality","ance","ances","ence","ences"
}

# Block splits when the left side is a common standalone prefix
NO_SPLIT_LEFT_PREFIXES = {
    "inter","intra","ultra","super","hyper","hypo","over","under","non","anti","auto",
    "bio","geo","photo","electro","thermo","hydro","micro","macro","nano","tele","trans",
    "sub","pre","post","out","cross"
}

# Unicode-aware "letters only": [^\W\d_] matches letters in Python's 're'
LETTER_CLASS = r"[^\W\d_]"
ALPHA_TOKEN_RE = re.compile(rf"^{LETTER_CLASS}+$", re.UNICODE)
ENDS_WORD_HYPHEN_RE = re.compile(rf"{LETTER_CLASS}{{2,}}-\s*$", re.UNICODE)



### Helper functions ###

def strip_diacritics(s: str) -> str:
    """Stripping diacritics."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def plausible_word(w: str, thr: float = ZIPF_THRESHOLD) -> bool:
    """Identifying the frequency of stripped word based on default threshold value."""
    lw = strip_diacritics(w).lower()
    return zipf_frequency(lw, "en") >= thr

def plausible_word_strict(w: str, thr: float = ZIPF_THRESHOLD) -> bool:
    """Identifying the frequency of stripped word based on higher threshold value."""
    lw = strip_diacritics(w).lower()
    return zipf_frequency(lw, "en") >= (thr + 0.2)

def looks_alpha(s: str) -> bool:
    """Checking if the word is alphanumeric."""
    return bool(s) and s.isalpha()

def concat_overlap(a: str, b: str) -> str:
    """Checking if the identified word is actual word."""
    if not a or not b:
        return a + b
    return (a + b[1:]) if a[-1].lower() == b[0].lower() else (a + b)

def try_morph_merge(left: str, right: str) -> Optional[str]:
    """Checking word morhphology."""
    L = left
    R = right.lower()
    if R == "ted" and L.endswith("t"):
        cand = L + "ed"
        if plausible_word(cand):
            return cand
    for suf in ("ed", "ing", "s", "es", "er", "est", "ly", "ment", "tion"):
        if R == suf:
            cand = L + suf
            if plausible_word(cand):
                return cand
    if R == "t":
        cand = L + "t"
        if plausible_word(cand):
            return cand
    return None

def stitch_spacebroken_lines(lines: List[str],
                             zipf_thr: float = ZIPF_THRESHOLD,
                             min_len: int = MIN_COMBINED_LEN) -> List[str]:
    """Stiching words broken by space."""
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            line_core = line.rstrip("\n")
            m_tail = re.search(rf"({LETTER_CLASS}+)\s*$", line_core)
            m_head = re.match(rf"\s*({LETTER_CLASS}+)", next_line)
            if m_tail and m_head:
                left = m_tail.group(1)
                right = m_head.group(1)
                starts_lower = right and right[0].islower()
                if starts_lower and len(left + right) >= min_len and plausible_word(left + right, zipf_thr):
                    merged_line = line_core[: m_tail.start(1)] + (left + right) + "\n"
                    remainder = next_line[m_head.end(0):]
                    out.append(merged_line + remainder)
                    i += 2
                    continue
        out.append(line)
        i += 1
    return out

def stitch_hyphenated_lines(lines: List[str],
                            zipf_thr: float = ZIPF_THRESHOLD,
                            min_len: int = MIN_COMBINED_LEN) -> List[str]:
    """Stiching words broken by end of a line."""
    stitched = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines) and ENDS_WORD_HYPHEN_RE.search(line):
            next_line = lines[i + 1]
            m_head = re.match(rf"\s*({LETTER_CLASS}+)", next_line)
            if m_head:
                left = ENDS_WORD_HYPHEN_RE.search(line).group(0).rstrip()[:-1]
                right = m_head.group(1)
                combo = left + right
                if len(combo) >= min_len and plausible_word(combo, zipf_thr):
                    base = ENDS_WORD_HYPHEN_RE.sub(left, line)
                    merged = base.rstrip("\n") + next_line[m_head.end(0):]
                    stitched.append(merged)
                    i += 2
                    continue
        stitched.append(line)
        i += 1
    return stitched

def normalize_spaces(s: str) -> str:
    """Space normalization in unicode."""
    s = s.replace("\u00AD", "")   # soft hyphen
    s = s.replace("\u200B", "")   # zero-width space
    s = s.replace("\uFEFF", "")   # BOM
    s = re.sub(r"[^\S\r\n]", " ", s)
    s = re.sub(r" {2,}", " ", s)
    return s

def merge_split_words_in_line(line: str,
                              min_len: int = MIN_COMBINED_LEN,
                              thr: float = ZIPF_THRESHOLD) -> str:
    """Merging words split across line."""
    def looks_alpha_token(tok: str) -> bool:
        return bool(tok) and looks_alpha(tok)

    SMALL_SUFFIXES = {
        "t","d","ed","ing","s","es","er","est","ly","ment","tion",
        "ness","ship","able","ible","ance","ence","ity","ation","ization","sion","al","ally"
    }

    parts = re.findall(r"\S+|\s+", line)
    if not parts:
        return line

    out = []
    i = 0
    while i < len(parts):
        if (
            i + 4 < len(parts)
            and parts[i].strip() and parts[i+1].isspace()
            and parts[i+2].strip() and parts[i+3].isspace()
            and parts[i+4].strip()
        ):
            left, sep1, mid, sep2, suf = parts[i], parts[i+1], parts[i+2], parts[i+3], parts[i+4]
            if looks_alpha_token(left) and looks_alpha_token(mid) and looks_alpha_token(suf) and suf.lower() in SMALL_SUFFIXES:
                stem = concat_overlap(left, mid)
                if len(stem) + len(suf) >= min_len:
                    m = try_morph_merge(stem, suf)
                    if m and plausible_word(m, thr):
                        out.append(m); i += 5; continue
                combo = concat_overlap(stem, suf)
                if len(combo) >= min_len and plausible_word(combo, thr):
                    out.append(combo); i += 5; continue

        if (
            i + 2 < len(parts)
            and parts[i].strip() and parts[i+1].isspace() and parts[i+2].strip()
        ):
            left, sep, right = parts[i], parts[i+1], parts[i+2]
            if looks_alpha_token(left) and looks_alpha_token(right):
                combo = concat_overlap(left, right)
                if len(combo) >= min_len and plausible_word(combo, thr):
                    out.append(combo); i += 3; continue
                m = try_morph_merge(left, right)
                if m and len(m) >= min_len and plausible_word(m, thr):
                    out.append(m); i += 3; continue

        out.append(parts[i]); i += 1

    return "".join(out)

def try_best_two_way_split(token: str, thr: float = ZIPF_THRESHOLD) -> Optional[Tuple[str, str]]:
    """
    Consider a split only if:
      - token is letters and len >= SPLIT_MIN_LEN
      - EITHER zipf(full) < 2.0 (rare) OR margin >= 2.5 (very strong)
      - parts len >= MIN_PART_LEN
      - left NOT in NO_SPLIT_LEFT_PREFIXES
      - right NOT in NO_SPLIT_RIGHT_SUFFIXES
      - each part clears thr+0.2
    """
    if not looks_alpha(token) or len(token) < SPLIT_MIN_LEN:
        return None

    tok_norm = strip_diacritics(token).lower()
    z_full = zipf_frequency(tok_norm, "en")

    best = None
    best_margin = 0.0

    for k in range(MIN_PART_LEN, len(token) - MIN_PART_LEN + 1):
        left = token[:k]
        right = token[k:]
        if not (looks_alpha(left) and looks_alpha(right)):
            continue

        l_norm = strip_diacritics(left).lower()
        r_norm = strip_diacritics(right).lower()

        # hard blocks
        if l_norm in NO_SPLIT_LEFT_PREFIXES:
            continue
        if r_norm in NO_SPLIT_RIGHT_SUFFIXES:
            continue

        if not (plausible_word_strict(left, thr) and plausible_word_strict(right, thr)):
            continue

        z_left = zipf_frequency(l_norm, "en")
        z_right = zipf_frequency(r_norm, "en")
        margin = (z_left + z_right) - z_full

        # allow if very rare OR margin is very strong
        if (z_full < 2.0) or (margin >= 2.5):
            if margin > best_margin:
                best_margin = margin
                best = (left, right)

    return best

def split_runons_in_line(line: str, thr: float = ZIPF_THRESHOLD) -> str:
    """SSpliting run ons in a line."""
    def repl_token(tok: str) -> str:
        had_dot = tok.endswith(".")
        core = tok[:-1] if had_dot else tok  # strip trailing dot for processing

        if len(core) >= SPLIT_MIN_LEN:
            split = try_best_two_way_split(core, thr)  # should return (left, right) or None
            if split:
                joined = f"{split[0]} {split[1]}"
                return joined + ("." if had_dot else "")

        # no split (or too short) will return original token unchanged
        return tok

    parts = re.findall(r"\S+|\s+", line)
    return "".join(p if p.isspace() else repl_token(p) for p in parts)

def preprocess_text(file_in: str | Path,
                    output_name: str,
                    zipf_thr: float = ZIPF_THRESHOLD,
                    min_len: int = MIN_COMBINED_LEN) -> Path:
    """Denoising the text file"""
    file_in = Path(file_in)

    if not file_in.exists():
        raise FileNotFoundError(f"Input file not found: {file_in}")

    output_file = file_in.with_name(output_name)

    text = file_in.read_text(encoding="utf-8", errors="replace")

    text = normalize_spaces(text)
    text = strip_diacritics(text)

    lines = text.splitlines(keepends=True)
    lines = stitch_hyphenated_lines(lines, zipf_thr, min_len)
    lines = stitch_spacebroken_lines(lines, zipf_thr, min_len)

    out_lines = []
    for ln in lines:
        ln = merge_split_words_in_line(ln, min_len=min_len, thr=zipf_thr)
        ln = split_runons_in_line(ln, thr=zipf_thr)
        out_lines.append(ln)

    processed_text = "".join(out_lines)
    output_file.write_text(processed_text, encoding="utf-8")

    return output_file
