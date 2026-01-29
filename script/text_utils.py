# text_utils.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import pandas as pd
import regex as re
import gdown



def clean_spaces(text: object) -> str:
    """Remove all whitespace characters from a value (safe for NaN/None)."""
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", "", str(text))


def clean_spaces_startend(text: object) -> str:
    """Strip leading/trailing whitespace (safe for NaN/None)."""
    if pd.isna(text):
        return ""
    s = str(text)
    s = re.sub(r"^\s+", "", s)
    s = re.sub(r"\s+$", "", s)
    return s


def extract_access(link: object) -> str:
    """Extract 'open' or 'closed' from the Link field; returns 'unknown' if missing."""
    if pd.isna(link):
        return "unknown"
    link = str(link)
    # if pattern doesn't match, return 'unknown' rather than the full string
    m = re.search(r"(open|closed)", link, flags=re.IGNORECASE)
    return m.group(1).lower() if m else "unknown"


def clean_links(link: object) -> str:
    """Clean up the original HTML-ish link snippet."""
    if pd.isna(link):
        return ""
    s = str(link)
    s = re.sub(r"\<a href=", "", s)
    s = re.sub(r"target+.{10,110}", "", s)
    s = re.sub(r'^\"+', "", s)
    return s


def clean_gdocs(link: str) -> str:
    """Extract gdocs file id from IRRI Google Docs link."""
    s = re.sub(r"https://docs.google.com/a/irri.org/file/d/", "", link)
    s = re.sub(r"/view+.{10,110}", "", s)
    # original code trimmed last 7 chars; keep behavior, but guard length
    return s[:-7] if len(s) >= 7 else s


def clean_dois(link: object) -> str:
    """Remove whitespace/quotes from DOI-like links."""
    if pd.isna(link):
        return ""
    s = str(link)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\"", "", s)
    return s


def process_link(link: object) -> Tuple[str, str]:
    """Return (snippet, source_type) where source_type is 'gdocs', 'doi', or 'none'.""" 
    if pd.isna(link):
        return "", "none"
    s = str(link)
    if re.search(r"google", s, flags=re.IGNORECASE):
        return clean_gdocs(s), "gdocs"
    return clean_dois(s), "doi"


def african_papers(title: object) -> str:
    """Tag titles containing 'africa' as african, else other."""
    if pd.isna(title):
        return "other"
    return "african" if re.search(r"africa", str(title), flags=re.IGNORECASE) else "other"


def clean_title(title: object, max_len: int = 120) -> str:
    """Filename-safe-ish title slug (keeps your existing behavior, just condensed)."""
    if pd.isna(title):
        return "untitled"

    s = str(title)

    # replace many punctuations/spaces with underscores
    s = re.sub(r"[ \:\`\'\,\;\(\)\/\-\+\"]", "_", s)
    s = re.sub(r"[&\.\´\–\?]", "_", s)
    s = re.sub(r"pdf", "", s, flags=re.IGNORECASE)
    s = re.sub(r"_+", "_", s)

    return s[:max_len]



def download_pdf(row: pd.Series, out_dir: str | Path = "dataset", sleep_s: float = 2.0) -> None:
    """Download a PDF for a single entry, when Access == 'open' and Source contains 'gdocs'.""" 
    access = row.get("Access")
    if pd.isna(access) or access != "open":
        return

    title = clean_title(row.get("Article"))
    typ = row.get("Literature_type")
    year = row.get("Year")

    title_combo = f"{year}__{typ}__{title}"
    out_path = Path(out_dir) / f"{title_combo}.pdf"
    url = row.get("Link_snippet")

    # gdown expects an ID for gdrive
    if re.search(r"gdocs", str(row.get("Source", "")), flags=re.IGNORECASE):
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            gdown.download(id=str(url), output=str(out_path), quiet=False, fuzzy=True)
        except OSError as err:
            print("OS error:", err)
        except Exception as e:
            print(f"Unexpected error during download: {e}")
    else:
        print("download error: not suitable")

    time.sleep(sleep_s)
