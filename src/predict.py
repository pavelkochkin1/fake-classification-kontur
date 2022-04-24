import sys
import argparse
import yaml
from pathlib import Path

import torch
from utils import get_device, get_project_root
import pandas as pd
import nltk
from nltk.corpus import stopwords
from torch.utils.data import DataLoader

from data import preproccess_corpus, TextClassificationDataset
from evaluating import make_prediction
 
def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument ('-f', '--file')
    parser.add_argument ('-t', '--text')

    return parser
 
 
if __name__ == '__main__':
    project_root: Path = get_project_root()
    with open(str(project_root / "config.yml")) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    model = torch.load("best_model.pth")
    model.to(get_device())
 
    if namespace.text is not None:
        test_df = pd.DataFrame(
            {
                params['data']['text_field_name']: namespace.text,
                params['data']['label_field_name']: 0,
            }
        )

    if namespace.file is not None:
        test_df = pd.read_csv(
            namespace.file,
            sep=params['data']['separator'],
        )

    if params['preprocessing']['rm_stopwords'] == True:
        nltk.download('stopwords')
        stop_words = stopwords.words('russian')
    else:
        stop_words = None

    test_df[params['data']['text_field_name']] = preproccess_corpus(
        df=test_df,
        text_column=params['data']['text_field_name'],
        stopwords=stop_words,
        lemmatize=params['preprocessing']['lemmatization'],
    )

    test_dataset = TextClassificationDataset(
        texts=test_df[params["data"]["text_field_name"]].values.tolist(),
        labels=test_df[params["data"]["label_field_name"]].values,
        max_seq_length=params["model"]["max_seq_length"],
        model_name=params["model"]["model_name"],
    )

    test_loader = {
        "test": DataLoader(
            dataset=test_dataset,
            batch_size=params["training"]["batch_size"],
            shuffle=False,
        )
    }
    model = torch.load("best_model.pth")
    model.to(get_device())

    test_df[params['data']['label_field_name']] = make_prediction(
        loader=test_loader,
        device=get_device(),
        model=model,
    )
    if namespace.file is not None:
        test_df[[params['data']['text_field_name'], params['data']['label_field_name']]].to_csv(
            namespace.file[:-4] + "_pred.tsv",
        )
        print(f"Prediction in {namespace.file[:-4] + '_pred.tsv'}")
    else:
        test_df[[params['data']['text_field_name'], params['data']['label_field_name']]].to_csv(
            "text_pred.tsv",
        )

    