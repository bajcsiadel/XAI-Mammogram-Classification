import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_model_w_condition(
    model, model_dir, model_name, accu, target_accu, log=print
):
    """
    model: this is not the multigpu model
    """
    if accu > target_accu:
        log(f"INFO: \t\t\tabove {target_accu:.2%}")
        # torch.save(
        #     obj=model.state_dict(),
        #     f=os.path.join(model_dir, f'{model_name}{accu:.4f}.pth')
        # )
        torch.save(
            obj=model,
            f=os.path.join(model_dir, f"{model_name}-{accu:.4f}.pth"),
        )


def save_image(fname, arr):
    if np.max(arr) > 1:
        arr = arr / 255.0
    if arr.shape[-1] == 1:
        plt.imsave(
            fname=fname,
            arr=arr.squeeze(axis=2),
            cmap="gray",
        )
    else:
        plt.imsave(
            fname=fname,
            arr=arr,
        )
