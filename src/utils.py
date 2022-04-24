from pathlib import Path
import re
from typing import List, Union

from tqdm import tqdm
import pymorphy2
import pandas as pd
import torch


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def clear_text(
    text: str, 
    just_letters: bool = False,
) -> str:
    """
    Converts a line to lowercase or delete everything except letters.
    Args:
        text: str - a string to clear

    Returns:
        string in lowercase
    """

    if just_letters:
        text = re.sub(r'[^А-яЁё]+', ' ', text).lower().replace("ё", "е")
        return ' '.join(text.split())

    text = text.lower().replace("ё", "е")
    return text.strip()
    

def clean_stop_words(
    text: str, 
    stopwords: List[str],
) -> str:
    """
    Removes stop words from the string.
    Args:
        text: str - a string to remove stopwords
        stopwords: List[str] - a list with stopwords

    Returns:
        string without stopwords
    """

    text = [word for word in text.split() if word not in stopwords]
    return " ".join(text)

lemmatizer = pymorphy2.MorphAnalyzer()

def lemmatize(
    corpus: List[str],
) -> List[str]:
    """
    Brings all the words for the texts from the corpus to the original form.
    Args:
        corpus: List[str]

    Returns:
        List[str]
    """

    result = list()

    for text in tqdm(corpus):
        lem_text = list()
        for word in text.split():
            lem_text.append(lemmatizer.parse(word)[0].normal_form)

        result.append(" ".join(lem_text))
        
    return result


def preproccess_corpus(
    df: Union[pd.Series, pd.DataFrame],
    text_column: Union[str, None] = None,
    stopwords: Union[List[str], None] = None,
    lemmatize: bool = True,
) -> pd.Series:
    """
    Full text preprocessing.

    Args:
        df: pd.Series or pd.DataFrame
        text_column: str if pd.DataFrame else None
        stopwords: List[str] if you need to remove stopwords else None
        lemmatize: True if you need lemmatization else False
    Returns:
        pd.Series
    """
    if type(df) == pd.DataFrame:
        df = df[text_column]

    result = list()
    for text in tqdm(df.to_list()):
        text = clear_text(text)
        if stopwords:
            text = clean_stop_words(text, stopwords)
        result.append(text)

    if lemmatize:
        result = lemmatize(result, lemmatizer)
    
    return pd.Series(result, index = df.index)

def get_device() -> str:
    """Videocard access check"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'