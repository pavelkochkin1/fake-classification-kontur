from torch.utils.data import DataLoader
from tqdm import tqdm
from model import BertForSequenceClassification
import numpy as np
from sklearn.metrics import classification_report, f1_score


def make_prediction(
    loader: DataLoader,
    device: str,
    model: BertForSequenceClassification,
) -> np.array:
    """
    Calculates predictions for loader.
    Args:
        loader: DataLoader - from TextClassifierDataset
        device: str - `cuda` or `cpu`
        model: BertForSequenceClassification - pretrained model
    Returns:
        array with predictions
        size: (bs, 1)
    """

    pred_labels = []

    for batch in tqdm(loader["test"]):
        inputs = batch['features'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = model(inputs, attention_mask).argmax(axis=1).cpu()
        pred_labels.append(output)

    pred_labels = np.concatenate(pred_labels, axis=0)

    return pred_labels


def classification_rep(
    loader: DataLoader,
    device: str,
    model: BertForSequenceClassification,
):
    """
    Prints classification report from sklearn for validation data.
    Args:
        loader: DataLoader - with `valid` key
        device: str - `cuda` or `cpu`
        model: BertForSequenceClassification - pretrained model
    """
    pred_labels = []
    true_labels = []

    for batch in tqdm(loader["valid"]):
        inputs = batch['features'].to(device)
        labels = batch['targets'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = model(inputs, attention_mask).argmax(axis=1).cpu()
        pred_labels.append(output)
        true_labels.append(labels.cpu().numpy())

    true_labels = np.concatenate(true_labels, axis=0)
    pred_labels = np.concatenate(pred_labels, axis=0)

    print(f"\nf1_score: {f1_score(true_labels, pred_labels)}")
    print(classification_report(true_labels, pred_labels))