from dotenv import load_dotenv
load_dotenv()

from torch.utils.data import DataLoader
from ProtoPNet.dataset.metadata import DATASETS
from ProtoPNet.dataset.dataloaders import CustomVisionDataset

# --------------------------------------------- #
import argparse

parser = argparse.ArgumentParser(__file__, "Define the mean and std of a given dataset.")

parser.add_argument("--dataset", type=str, required=True, help="Dataset for which to define the mean and std.")
# parser.add_argument("--")

args = parser.parse_args()
# --------------------------------------------- #

assert args.dataset in DATASETS.keys(), f"Dataset must be one of {DATASETS.keys()}, not {args.dataset}"

DS_META = DATASETS[args.dataset]

for version in DS_META.VERSIONS.values():
    DS_META.IMAGE_DIR = version.DIR
    print(DS_META.NAME, version.NAME)
    dataset = CustomVisionDataset(DS_META, "normal_vs_abnormal", subset="train")

    loader = DataLoader(dataset, batch_size=len(dataset), num_workers=0, shuffle=False)

    images, _ = next(iter(loader))

    mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])
    print(f"MEAN={mean.numpy()[0]:.4f},\nSTD={std.numpy()[0]:.4f},\n")
