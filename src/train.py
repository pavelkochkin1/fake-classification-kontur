from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from catalyst.dl import SupervisedRunner
from catalyst.callbacks import (
    AccuracyCallback,
    OptimizerCallback,
    PrecisionRecallF1SupportCallback,
)
from catalyst.utils import prepare_cudnn, set_global_seed

from data import get_ready_data
from model import BertForSequenceClassification
from utils import get_device, get_project_root, get_device
from evaluating import make_prediction, classification_rep


# loading config params
project_root: Path = get_project_root()
with open(str(project_root / "config.yml")) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# reproducibility
set_global_seed(params["general"]["seed"])
prepare_cudnn(deterministic=True)


# read and process data
train_val_loaders, test_loaders = get_ready_data(params)
print("Read and processed data...")


# initialize the model
model = BertForSequenceClassification(
    pretrained_model_name=params["model"]["model_name"],
    num_classes=params["model"]["num_classes"],
    dropout=0.3,
)


# specify criterion for the classification task, optimizer and scheduler
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    params=model.parameters(), 
    lr=float(params["training"]["learn_rate"]),
)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer=optimizer, 
    milestones=[2,4], 
)


# here we specify that we pass masks to the runner. So model's forward method will be called with
# these arguments passed to it.
runner = SupervisedRunner(input_key=("features", "attention_mask"))


# finally, training the model with Catalyst
print("Started training...")
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=train_val_loaders,
    callbacks=[
        AccuracyCallback(
            num_classes=int(params["model"]["num_classes"]), 
            input_key="features", 
            target_key='targets',
        ),
        OptimizerCallback(metric_key="loss"),
        PrecisionRecallF1SupportCallback(
            input_key="logits",
            target_key="targets", 
            num_classes=2,
        ),
    ],
    logdir=params["training"]["log_dir"],
    num_epochs=int(params["training"]["num_epochs"]),
    load_best_on_end=True,
    verbose=True,
)


# check the score on validation data
classification_rep(
    train_val_loaders,
    device=get_device(),
    model=runner.model,
)


# saving model
print("Model saving...")
torch.save(runner.model, "best_model.pth")


# test data prediction 
test_df = pd.read_csv(
    Path(params['data']['path_to_data']) / params['data']['test_filename'],
    sep=params['data']['separator'],
)

test_df[params['data']['label_field_name']] = make_prediction(
    loader=test_loaders, 
    device=get_device(), 
    model=runner.model,
)

test_df[[params['data']['text_field_name'], params['data']['label_field_name']]].to_csv(
    params['data']['path_to_test_pred_scores'],
)
print(f"Test data prediction in {params['data']['path_to_test_pred_scores']}")