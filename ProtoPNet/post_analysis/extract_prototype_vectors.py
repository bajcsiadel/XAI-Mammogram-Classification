import os
import re

import numpy as np
import torch
from config.settings import base_architecture, experiment_run
from ProtoPNet.util.helpers import makedir

epoch = 278
search_dir = os.path.join(
    os.getcwd(), "saved_models", base_architecture, experiment_run, "model"
)
save_dir = os.path.join(search_dir, "prototype_vectors")
model_regex = re.compile(r"^[0-9]+nopush.*\.pth$")

torch.cuda.set_device(1)
makedir(save_dir)

if __name__ == "__main__":
    for file in os.listdir(search_dir):
        if model_regex.match(file):
            load_path = os.path.join(search_dir, file)
            save_path = os.path.join(save_dir, file.replace(".pth", ".csv"))
            model = torch.load(
                load_path, map_location=lambda storage, _loc: storage
            )
            prototypes_matrix = (
                torch.squeeze(model.prototype_vectors).cpu().detach().numpy()
            )
            np.savetxt(save_path, prototypes_matrix, delimiter=",")
