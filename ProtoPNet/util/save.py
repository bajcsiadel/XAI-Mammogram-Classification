import os

import matplotlib.pyplot as plt
import torch


def save_model_w_condition(
    model, model_dir, model_name, accu, target_accu, log=print
):
    """
    model: this is not the multigpu model
    """
    if accu > target_accu:
        log("\tabove {0:.2f}%".format(target_accu * 100))
        # torch.save(
        #     obj=model.state_dict(),
        #     f=os.path.join(model_dir, f'{model_name}{accu:.4f}.pth')
        # )
        torch.save(
            obj=model,
            f=os.path.join(model_dir, f"{model_name}{accu:.4f}.pth"),
        )


def save_image(fname, arr):
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
