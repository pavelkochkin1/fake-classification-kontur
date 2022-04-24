import logging
from pathlib import Path
from typing import List, Mapping, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from catalyst.utils import set_global_seed
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords

from utils import preproccess_corpus

class TextClassificationDataset(Dataset):
    """
    Wrapper around Torch Dataset to perform text classification.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[str] = None,
        label_dict: Mapping[str, int] = None,
        max_seq_length: int = 32,
        model_name: str = "SkolkovoInstitute/russian_toxicity_classifier",
    ):
        """
        Args:
            texts: (List[str]) - a list with texts to classify or to train the
                classifier
            labels: List[str] - a list with classification labels (optional)
            max_seq_length: (int) - maximal sequence length in tokens
            model_name: (str) - transformer model name
        """
        self.texts = texts
        self.labels = labels
        self.label_dict = label_dict
        self.max_seq_length = max_seq_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.getLogger("transformers.tokenization_utils").setLevel(logging.FATAL)

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """

        return len(self.texts)
    
    def __getitem__(self, index) -> Mapping[str, torch.Tensor]:
        """Gets element of the dataset by index
        Args:
            index: (int) - index of the element in the dataset
        Returns:
            Element by index
        """

        text = self.texts[index]

        # a dictionary with `input_ids` and `attention_mask` as keys
        output_dict = self.tokenizer.encode_plus(
            text,
            add_special_tokens = True,
            padding = "max_length",
            max_length = self.max_seq_length,
            return_tensors = "pt",
            truncation = True,
            return_attention_mask = True,
        )

        # for Catalyst, there needs to be a key called `features`
        output_dict["features"] = output_dict["input_ids"].squeeze(0)
        del output_dict["input_ids"]
    
        if self.labels is not None:
            output_dict["targets"] = self.labels[index]
        
        return output_dict
    

def get_ready_data(params: dict) -> Tuple[dict, dict]:
    """
    A function that reads data from CSV files, preprocess text field in data, 
    creates PyTorch datasets and data loaders. 

    Args:
        params: dict - a dictionary read from the config.yml file
    Returns:
        A tuple with 2 dictionaries
    """
    # reading CSV files to Pandas dataframes
    train_df = pd.read_csv(
        Path(params['data']['path_to_data']) / params['data']['train_filename'],
        sep=params['data']['separator'],
    )
    if params['data']['valid_filename'] == 'None':
        train_df, valid_df = train_test_split(
            train_df,
            test_size=0.1, 
            random_state=params['general']['seed'],
        )
    else:
        valid_df = pd.read_csv(
            Path(params['data']['path_to_data']) / params['data']['valid_filename'],
            sep=params['data']['separator'],
        )
    test_df = pd.read_csv(
        Path(params['data']['path_to_data']) / params['data']['test_filename'],
        sep=params['data']['separator'],
    )

    if params['preprocessing']['rm_stopwords'] == True:
        nltk.download('stopwords')
        stop_words = stopwords.words('russian')
    else:
        stop_words = None

    train_df[params['data']['text_field_name']] = preproccess_corpus(
        df=train_df,
        text_column=params['data']['text_field_name'],
        stopwords=stop_words,
        lemmatize=params['preprocessing']['lemmatization'],
    )

    valid_df[params['data']['text_field_name']] = preproccess_corpus(
        df=valid_df,
        text_column=params['data']['text_field_name'],
        stopwords=stop_words,
        lemmatize=params['preprocessing']['lemmatization'],
    )
    print(valid_df)

    test_df[params['data']['text_field_name']] = preproccess_corpus(
        df=test_df,
        text_column=params['data']['text_field_name'],
        stopwords=stop_words,
        lemmatize=params['preprocessing']['lemmatization'],
    )
    

    # creating PyTorch Datasets
    train_dataset = TextClassificationDataset(
        texts=train_df[params["data"]["text_field_name"]].values.tolist(),
        labels=train_df[params["data"]["label_field_name"]].values,
        max_seq_length=params["model"]["max_seq_length"],
        model_name=params["model"]["model_name"],
    )

    valid_dataset = TextClassificationDataset(
        texts=valid_df[params["data"]["text_field_name"]],
        labels=train_df[params["data"]["label_field_name"]].values,
        max_seq_length=params["model"]["max_seq_length"],
        model_name=params["model"]["model_name"],
    )

    test_dataset = TextClassificationDataset(
        texts=test_df[params["data"]["text_field_name"]].values.tolist(),
        labels=test_df[params["data"]["label_field_name"]].values,
        max_seq_length=params["model"]["max_seq_length"],
        model_name=params["model"]["model_name"],
    )

    set_global_seed(params["general"]["seed"])

    # creating PyTorch data loaders and placing them in dictionaries (for Catalyst)
    train_val_loaders = {
        "train": DataLoader(
            dataset=train_dataset,
            batch_size=params["training"]["batch_size"],
            shuffle=True,
        ),
        "valid": DataLoader(
            dataset=valid_dataset,
            batch_size=params["training"]["batch_size"],
            shuffle=False,
        ),
    }

    test_loaders = {
        "test": DataLoader(
            dataset=test_dataset,
            batch_size=params["training"]["batch_size"],
            shuffle=False,
        )
    }

    return train_val_loaders, test_loaders