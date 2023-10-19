import os
import torch
import argparse
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score, accuracy_score

from database.dataloader import (
    test_loader
)

from ProtoPNet.config.settings import (
    base_architecture,
    experiment_run,
)

'''
    Usage: python3 test_accuracy.py -model='' -gpuid 0 --backbone
    - model: model name (.pth file)
    - gpuid: GPU id 
    Calculates model's accuracy on test dataset
'''

def calculate_f1(model, dataloader):
    targets = []
    predictions = []
    for image, target in dataloader:
        with torch.no_grad():
            input = image.cuda()
            target = target.cuda()
            if backbone_only:
                output = model(input)
            else:
                output, _ = model(input)
            _, predicted = torch.max(output, dim=1) 
            targets.extend(target.cpu())
            predictions.extend(predicted.cpu())
    f1_macro = f1_score(targets, predictions, average='macro')
    f1_micro = f1_score(targets, predictions, average='micro')
    accuracy = accuracy_score(targets, predictions)
    return f1_macro, f1_micro, accuracy

experiment_dir = os.path.join(
    os.getcwd(), "saved_models", base_architecture, experiment_run
)
model_dir = os.path.join(experiment_dir, 'model')
batch_size = 32

parser = argparse.ArgumentParser()
parser.add_argument("-gpuid", type=str, default="0")
parser.add_argument("-model", type=str, required=True)
parser.add_argument("--backbone", action="store_true")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
backbone_only = args.backbone
model_file = args.model
model_name = '.'.join(model_file.split('.')[:-1])

model_path = os.path.join(model_dir, model_file)

model = torch.load(model_path)
mac, mic, acc = calculate_f1(model, test_loader)
print(mac, mic, acc)

