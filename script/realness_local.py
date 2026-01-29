# realness.py
# Import modules
import sys
import os
import time
import nltk
from nltk.corpus import words
import requests
import time
import xlsxwriter
import pandas as pd
import spacy
from spacy.util import is_package
from spacy.cli import download
from pathlib import Path
import numpy as np

__all__ = [
    "paper_real_words"
]

nlp = spacy.load("en_core_web_lg")

def paper_real_words(df):
    """Calculating the realness score."""
    
    def force_utf8(value):
        """Fixing encoding issues."""
        try:
            return value.encode('utf-8').decode('utf-8', errors='replace')
        except Exception:
            return "ENCODING_ERROR"

    def paper_word_processing(df):
        """Return sorted unique words."""
        df_new = df
        df_new_sorted = df_new.sort_values(by='Word')
        df_new_sorted['Word'] = df_new_sorted['Word'].astype(str)
        df_new_sorted['Word'] =  df_new_sorted['Word'].apply(force_utf8)
        return df_new_sorted['Word'].drop_duplicates().reset_index(drop=True)

    def to_lowercase(data):
        """Making everything lowercase."""
        if isinstance(data, list):  
            return [to_lowercase(item) for item in data]
        elif isinstance(data, str):  
            return data.lower()
        else:  
            return data
    
    word_set = words.words()
    lowercase_word_set = to_lowercase(word_set)
    
    def is_real_word_nltk(word):
        """Checking if word is found in dataset words"""
        if word.isnumeric():
            return True
        else:
            return word.lower() in lowercase_word_set
   
    def is_real_word_spacy(word):
        """Checking if word is found in dataset en_core_web_lg"""
        token = nlp.vocab[word]
        return not token.is_oov
    
    def check_word_in_pubmed(word, api_key=None):
        """Checking if word is found in pubmed database"""
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        
        
        params = {
            'db': 'pubmed',
            'term': word,
            'retmax': '1',
            'usehistory': 'y',
            'api_key': api_key,
            'retmode': 'xml',
            'maxdate': "2025-08-15"
        }
        response = requests.get(base_url, params=params)
        time.sleep(0.34)
        if response.status_code == 200:
            xml_data = response.text
            if "<Count>0</Count>" in xml_data:
                return False
            else:
                return True
        else:
            return False

    
    stats = pd.DataFrame([[np.nan, np.nan, np.nan]], columns=['nltk', 'spacy', 'pubmed']) 
    
    after_nltk = paper_word_processing(df)
    print('paper words at start',len(after_nltk))
    x_nltk = after_nltk.apply(lambda word: is_real_word_nltk(word))
    print(after_nltk)
    
    print('number of words left after nltk library', (x_nltk==False).sum())
    stats.at[0, 'nltk'] = (x_nltk==False).sum()
    
    print('words after nltk:')    
    after_spacy = after_nltk[x_nltk==False].reset_index(drop=True)
    print(after_spacy)
    
    x_spacy = after_spacy.apply(lambda word: is_real_word_spacy(word))
    print('number of words left after spacy library', (x_spacy==False).sum())
    stats.at[0, 'spacy'] = (x_spacy==False).sum()
    
    print('words after spacy:')
    after_pubmed = after_spacy[x_spacy==False].reset_index(drop=True)
    print(after_pubmed)
    
    x_pubmed = after_pubmed.apply(lambda word: check_word_in_pubmed(word))
    print('number of words left after pubmed ', (x_pubmed==False).sum())
    stats.at[0, 'pubmed'] = (x_pubmed==False).sum()
    
    x_final = after_pubmed[x_pubmed==False].reset_index(drop=True)
    realness = round(1 - ((x_pubmed==False).sum()/len(df)),2)
    return realness, x_final, stats

