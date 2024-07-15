import xai_mam.models.BagNet._model.explainable as expl 
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import feature, transform
from PIL import Image

import dataclasses as dc
import os
import sys
import typing as typ

import hydra
from dotenv import load_dotenv
from omegaconf import OmegaConf
from omegaconf import errors as conf_errors
from torch.utils.data import DataLoader

load_dotenv()
sys.path.append(os.getenv("PROJECT_ROOT"))

from xai_mam.dataset.dataloaders import my_collate_function
from xai_mam.utils import custom_pipe
from xai_mam.utils.config import config_store_
from xai_mam.utils.config._general_types.data import DatasetConfig
from xai_mam.utils.config.resolvers import add_all_custom_resolvers
from xai_mam.utils.log import ScriptLogger


@dc.dataclass
class Data:
    set: DatasetConfig


@dc.dataclass
class Config:
    data: Data
    dataset: dict[str, typ.Any]

def plot_heatmap(
    heatmap,
    original,
    ax,
    cmap="RdBu_r",
    percentile=99,
    dilation=0.5,
    alpha=0.25,
):
    """
    Plots the heatmap on top of the original image
    (which is shown by most important edges).

    Parameters
    ----------
    heatmap : Numpy Array of shape [X, X]
        Heatmap to visualise.
    original : Numpy array of shape [X, X, 3]
        Original image for which the heatmap was computed.
    ax : Matplotlib axis
        Axis onto which the heatmap should be plotted.
    cmap : Matplotlib color map
        Color map for the visualisation of the heatmaps (default: RdBu_r)
    percentile : float between 0 and 100 (default: 99)
        Extreme values outside of the percentile range are clipped.
        This avoids that a single outlier dominates the whole heatmap.
    dilation : float
        Resizing of the original image. Influences the edge detector and
        thus the image overlay.
    alpha : float in [0, 1]
        Opacity of the overlay image.

    """
    if len(heatmap.shape) == 3:
        heatmap = np.mean(heatmap, 0)

    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, heatmap.shape[1], dx)
    yy = np.arange(0.0, heatmap.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_original = plt.get_cmap("Greys_r")
    cmap_original.set_bad(alpha=0)
    overlay = None
    if original is not None:
        # Compute edges (to overlay to heatmaps later)
        original_greyscale = (
            original
            if len(original.shape) == 2
            else np.mean(original, axis=-1)
        )
        in_image_upscaled = transform.rescale(
            original_greyscale,
            dilation,
            mode="constant",
            # multichannel=False, # no longer supported!
            anti_aliasing=True,
        )
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges

    abs_max = np.percentile(np.abs(heatmap), percentile)
    abs_min = abs_max

    ax.imshow(
        heatmap,
        extent=extent,
        interpolation="none",
        cmap=cmap,
        vmin=-abs_min,
        vmax=abs_max,
    )
    if overlay is not None:
        ax.imshow(
            overlay,
            extent=extent,
            interpolation="none",
            cmap=cmap_original,
            alpha=alpha,
        )


def plot_heatmap_protopnet_style(
    heatmap, original, ax, filename, pred, true, actual, blur=False, perc=99
):
    if len(heatmap.shape) == 3:
        heatmap = np.mean(heatmap, 0)

    # cutting off large pixel values
    perc = np.percentile(abs(heatmap), perc)
    heatmap[heatmap > perc] = perc
    heatmap[heatmap < -perc] = -perc

    vmin = np.amin(heatmap)
    vmax = np.amax(heatmap)
    heatmap = (heatmap - vmin) / (vmax - vmin)

    heatmap = np.exp(7 * heatmap)  # scaling pixel intensities

    vmin = np.amin(heatmap)
    vmax = np.amax(heatmap)
    heatmap = (heatmap - vmin) / (vmax - vmin)

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    if blur:
        heatmap = cv2.blur(heatmap, (2, 2))
    heatmap = np.float32(heatmap) / 255
    # heatmap = heatmap[
    #     ..., ::-1
    # ]  # inverting last dimension??? - inverts colors
    overlayed_img = 0.5 * original + 0.3 * heatmap

    print(type(overlayed_img))
    print(overlayed_img[1,:].shape)

    # overlayed_img_transposed = np.transpose(overlayed_img, (2,0,1))
 
    im = Image.fromarray((overlayed_img * 255).astype(np.uint8))
    im.save(f"/home/annamari/tankstorage/ProtoPNet-Mammogram/xai_mam/models/BagNet/scripts/heatmaps/mias/heatmap_{filename}_{pred}_{true}_{actual}.png")

    ax.imshow(overlayed_img)


def generate_heatmap_pytorch(
    config, model, image, target, patchsize, padding="replication"
):
    """
    Generates high-resolution heatmap for a BagNet by decomposing the
    image into all possible patches and by computing the logits for
    each patch.

    Parameters
    ----------
    model : Pytorch Model
        This should be one of the BagNets.
    image : Numpy array of shape [1, 3, X, X]
        The image for which we want to compute the heatmap.
    target : int
        Class for which the heatmap is computed.
    patchsize : int
        The size of the receptive field of the given BagNet.

    """
    import torch

    with torch.no_grad():
        _, c, x, y = image.shape

        if padding == "zero":  # original code
            padded_image = np.zeros((c, x + patchsize - 1, y + patchsize - 1))
            padded_image[
                :,
                (patchsize - 1) // 2 : (patchsize - 1) // 2 + x,
                (patchsize - 1) // 2 : (patchsize - 1) // 2 + y,
            ] = image[0]
            input = torch.from_numpy(padded_image[None].astype(np.float32))
        elif padding == "replication":
            padder = torch.nn.ReplicationPad2d((patchsize - 1) // 2)
            input = padder(torch.from_numpy(image))
        else:
            input = torch.from_numpy(image)

        # extract patches
        patches = input.permute(0, 2, 3, 1)
        patches = patches.unfold(1, patchsize, 1).unfold(2, patchsize, 1)

        patches = patches.contiguous().view(
            (-1, config.data.set.image_properties.n_color_channels, patchsize, patchsize)
        )

        # compute logits for each patch
        logits_list = []

        for batch_patches in torch.split(patches, 10000):
            print(f"Batch: {batch_patches.shape}")
            logits = model(batch_patches.to(next(model.parameters()).device))
            logits = logits[:, target]
            logits_list.append(logits.data.cpu().numpy().copy())

        # logits = np.hstack(logits_list)
        logits = np.vstack(logits_list)

        delta = (patchsize - 1) if padding is None else 0

        return logits.reshape(
            (config.data.set.image_properties.width - delta, config.data.set.image_properties.height - delta)
        )


batch_size = 1  # todo: read this from config
patch_size = 17

@hydra.main(
    version_base=None,
    config_path=os.getenv("CONFIG_PATH"),
    config_name="script_define_mean_config",
)
def main_plot(cfg: Config):


    print(cfg)
        
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = expl.bagnet17(
        2, None, n_color_channels=1, pretrained=True
    )

    # loc = 'cuda:{}'.format(0)

    # checkpoint = torch.load(
    #   "/home/miafranc/bosch/bagnet/model_best_bagnet17-bosch.pth.tar",
    #   map_location=device)
    checkpoint = torch.load(
        "/home/annamari/tankstorage/ProtoPNet-Mammogram/runs/main/bagnet/2024-07-12/MIAS-bagnet17-preprocessed-benign_vs_malignant/150-150-data-augmentation@data.set.image_properties.augmentations.train=repeated_shape_and_color_26/16-18-29/checkpoints/1-50-96.0986.pth",
        map_location=device,
    )


    model.load_state_dict(checkpoint["state_dict"])

    print("checkpoint stats:")
    print(checkpoint.keys())
    print(checkpoint["accu"])

    # print("The model will be running on", device, "device")
    # # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    model.eval()

    dataset = hydra.utils.instantiate(cfg.dataset)

    loader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=0,
    shuffle=False,
    collate_fn=my_collate_function,
    )


    filenumber = 0

# for i, (images, target, filename) in enumerate(val_loader):
    for _, (images, target) in enumerate(loader):

        print(type(images))

        # list of tensors -> tensor of tensors of [B x C x W x H]
        images = torch.stack(images, dim=0)

        image = images.numpy()
       
        # TODO: read w and h from config instead
        perm = image.reshape(224, 224) * 255
        perm = perm.astype(np.uint8)

        im = Image.fromarray(perm)
        im.save(f"/home/annamari/tankstorage/ProtoPNet-Mammogram/xai_mam/models/BagNet/scripts/heatmaps/mias/original_{filenumber}.png")


        target_num = target.detach().cpu().numpy()[0]

        images = images.to(device, non_blocking=True)
        output = model(images)

        pr = output.detach().cpu().numpy()[0].argmax()
        pred = np.append([], int(pr))

        # heatmap of the target class:
        heatmap_0 = generate_heatmap_pytorch(
        cfg, model, image, [target_num], patch_size, padding="replication"
    )

        # heatmap of the predicted class:
        # the heatmaps above can be the same
        heatmap_1 = generate_heatmap_pytorch(
        cfg, model, image, [pr], patch_size, padding="replication"
    )


    # plot heatmaps
        fig = plt.figure(figsize=(8, 4))

        original_image = image[0].transpose([1, 2, 0])

        ax = plt.subplot(211)

        plot_heatmap_protopnet_style(
        heatmap_0,
        original_image,
        ax,
        filenumber,
        pr,
        target_num,
        target_num,
        80,
        False,
    )
        plt.axis("off")

        ax = plt.subplot(212)

        plot_heatmap_protopnet_style(
        heatmap_1,
        original_image,
        ax,
        filenumber,
        pr,
        target_num,
        pr,
        80,
        False,
    )
        plt.axis("off")

        plt.savefig(
        f"./heatmaps/mias/{filenumber}_plot_{target_num}_{pr}.png"
    )

        filenumber = filenumber + 1


add_all_custom_resolvers()
config_store_.store(name="_config_validation", node=Config)
config_store_.store(name="_data_validation", group="data", node=Data)
config_store_.store(name="_data_set_validation", group="data/set", node=DatasetConfig)
main_plot()
