import argparse
import os

import pandas as pd
import torch
from database.dataloader import RoadTypeDetectionDatasetWithPath
from ProtoPNet.config.settings import base_architecture, experiment_run
from ProtoPNet.util.helpers import makedir

experiment_dir = os.path.join(
    os.getcwd(), "saved_models", base_architecture, experiment_run
)
model_dir = os.path.join(experiment_dir, "model")
batch_size = 32

test_dataset = RoadTypeDetectionDatasetWithPath(
    test=True,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)

if __name__ == "__main__":
    '''
        Usage:  python extract_misclassified_images.py -model='' -gpuid 0
        - model: name of the model (.pth file)
        - gpuid: GPU id
        Extracts the misclassified images and save their paths, correct labels and predictions
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpuid", nargs=1, type=str, default="0")
    parser.add_argument("-model", nargs=1, type=str)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid[0]
    model_name = args.model[0]

    model = ".".join(model_name.split(".")[:-1])
    save_dir = os.path.join(experiment_dir, model, "misclassification")
    makedir(save_dir)
    model = torch.load(os.path.join(model_dir, model_name))
    model = model.cuda()

    df = pd.DataFrame(columns=["image_path", "true_class", "predicted_class"])

    for image, label, path in test_loader:
        input = image.cuda()
        target = label.cuda()

        output, _ = model(input)

        _, predicted = torch.max(output.data, 1)
        target = label.cpu().numpy()
        prediction = predicted.cpu().numpy()
        for i in range(len(target)):
            true_class = target[i]
            pred = prediction[i]
            if pred != true_class:
                df.loc[len(df.index)] = [path[i], true_class, pred]
    df.to_csv(os.path.join(save_dir, "misclassified_images.csv"), index=False)
