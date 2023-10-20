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

test_dataset = RoadTypeDetectionDatasetWithPath(subset="test")

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)

if __name__ == "__main__":
    '''
        Usage: python save_predictions.py -model='' -gpuid 0 --backbone
        - model: model name (.pth file)
        - gpuid: GPU id 
        Saves the model's weights and predictions on the test dataset
    '''
    # python post_analysis/save_predictions.py -model="8nopush0.8186.pth" -gpuid 2 --backbone
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpuid", type=str, default="0")
    parser.add_argument("-model", type=str, required=True)
    parser.add_argument("--backbone", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    backbone_only = args.backbone

    model_file = args.model
    model_name = ".".join(model_file.split(".")[:-1])

    model_path = os.path.join(model_dir, model_file)
    save_dir = os.path.join(experiment_dir, model_name)
    makedir(save_dir)

    model = torch.load(model_path)
    last_layer = model.state_dict()["last_layer.weight"]
    df = pd.DataFrame(columns=["c1", "c2", "c3"])
    df["c1"] = last_layer[0].cpu().numpy()
    df["c2"] = last_layer[1].cpu().numpy()
    df["c3"] = last_layer[2].cpu().numpy()
    df.to_csv(os.path.join(save_dir, "last_layer_weights.csv"), index=False)

    df = pd.DataFrame(
        columns=[
            "image_path",
            "true_class",
            "predicted_class",
            "act1",
            "act2",
            "act3",
        ]
    )
    with torch.no_grad():
        for image, label, path in test_loader:
            input = image.cuda()
            target = label.cuda()

            if backbone_only:
                output = model(input)
            else:
                output, _ = model(input)
            _, predicted = torch.max(output.data, 1)
            probabilities = output.data

            target = label.cpu().numpy()
            prediction = predicted.cpu().numpy()

            for i in range(len(target)):
                prob = probabilities[i].cpu().numpy()
                df.loc[len(df.index)] = [
                    path[i],
                    target[i],
                    prediction[i],
                    prob[0],
                    prob[1],
                    prob[2],
                ]
    df.to_csv(os.path.join(save_dir, "results.csv"), index=False)
