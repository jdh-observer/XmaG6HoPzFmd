# text_processing.py
import pandas as pd
import re
import os
from pathlib import Path
from os import listdir
from sklearn.feature_extraction.text import CountVectorizer

__all__ = [
    "read_text",
    "wfa"
]

def read_text(file_path):
    """Reading a txt file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def wfa(file_in: str | Path) -> pd.DataFrame:
    """Compute word frequencies for a single text file.""" 
    file_in = Path(file_in)

    if not file_in.exists():
        raise FileNotFoundError(f"Input file not found: {file_in}")

    text = read_text(file_in) 

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([text])

    df = pd.DataFrame({
        "Word": vectorizer.get_feature_names_out(),
        "Frequency": X.toarray().ravel(),
    })

    return df

