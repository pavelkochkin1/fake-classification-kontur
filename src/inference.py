from torch.utils.data import DataLoader
from tqdm import tqdm
from model import BertForSequenceClassification
import numpy as np

def make_prediction(
    loader: DataLoader,
    device: str,
    model: BertForSequenceClassification,
    ) -> np.array:

    pred_labels = []

    for batch in tqdm(loader["test"]):
        inputs = batch['features'].to(device)
        # labels = batch['targets'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = model(inputs, attention_mask).argmax(axis=1).cpu()
        pred_labels.append(output)

    pred_labels = np.concatenate(pred_labels, axis=0)

    return pred_labels

    