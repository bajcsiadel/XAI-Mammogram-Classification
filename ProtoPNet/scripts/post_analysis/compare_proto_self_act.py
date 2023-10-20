import os

import numpy as np
from ProtoPNet.config.settings import (
    backbone_only,
    base_architecture,
    experiment_run,
)

EPOCH = 10
epoch_dir = os.path.join(
    "./saved_models/",
    base_architecture,
    experiment_run + ("_backbone" if backbone_only else ""),
    "img",
    f"epoch-{EPOCH}",
)


def main():
    proto_original_img_idx = np.load(
        os.path.join(epoch_dir, f"bb{EPOCH}.npy")
    )[:, 0]

    for prototype_idx, original_img_idx in enumerate(proto_original_img_idx):
        heatmap = np.load(
            os.path.join(
                epoch_dir,
                f"prototype-self-act{str(prototype_idx)}.npy",
            )
        )
        prototype_idx, original_img_idx, heatmap.shape


if __name__ == "__main__":
    main()
