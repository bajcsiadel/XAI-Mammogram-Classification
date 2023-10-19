import os

DATA_DIR = os.getenv("DATASET_LOCATION")
assert DATA_DIR is not None, "Please set the environment variable DATASET_LOCATION in .env file"

empty_dataset = {
    "DATASET_DIR": "",
    "ORIGINAL_IMAGE_DIR": "",
    "PREPROCESSED_IMAGE_DIR": "",
    "SPLIT_FILE": "",
    "METADATA_FILE": "",
    "IMAGE_SHAPE": (0, 0),
    "COLOR_CHANNELS": 0,
    "CLASSES": [],
    "NUMBER_OF_CLASSES": 0,
}

# dataset configs
DATASETS = {
    "MIAS": empty_dataset.copy(),
    "DDSM": empty_dataset.copy(),
}

# MIAS
DATASETS["MIAS"]["DATASET_DIR"] = os.path.join(DATA_DIR, "MIAS")
DATASETS["MIAS"]["ORIGINAL_IMAGE_DIR"] = os.path.join(DATASETS["MIAS"]["DATASET_DIR"], "pngs")
DATASETS["MIAS"]["PREPROCESSED_IMAGE_DIR"] = os.path.join(DATASETS["MIAS"]["DATASET_DIR"], "masked_images")
DATASETS["MIAS"]["SPLIT_FILE"] = os.path.join(DATASETS["MIAS"]["DATASET_DIR"], "split.npz")
DATASETS["MIAS"]["METADATA_FILE"] = os.path.join(DATASETS["MIAS"]["DATASET_DIR"], "data.csv")

DATASETS["MIAS"]["IMAGE_SHAPE"] = (1024, 1024)
DATASETS["MIAS"]["COLOR_CHANNELS"] = 1
DATASETS["MIAS"]["CLASSES"] = ["B", "M", "N"]
DATASETS["MIAS"]["NUMBER_OF_CLASSES"] = len(DATASETS["MIAS"]["CLASSES"])


# DDSM
DATASETS["DDSM"]["DATASET_DIR"] = os.path.join(DATA_DIR, "DDSM")
DATASETS["DDSM"]["ORIGINAL_IMAGE_DIR"] = os.path.join(DATASETS["DDSM"]["DATASET_DIR"], "images")
DATASETS["DDSM"]["PREPROCESSED_IMAGE_DIR"] = os.path.join(DATASETS["DDSM"]["DATASET_DIR"], "masked_images")
DATASETS["DDSM"]["SPLIT_FILE"] = os.path.join(DATASETS["DDSM"]["DATASET_DIR"], "split.npz")
DATASETS["DDSM"]["METADATA_FILE"] = os.path.join(DATASETS["DDSM"]["DATASET_DIR"], "data.csv")

DATASETS["DDSM"]["IMAGE_SHAPE"] = (1024, 1024)
DATASETS["DDSM"]["COLOR_CHANNELS"] = 1
DATASETS["DDSM"]["CLASSES"] = ["B", "M", "N"]
DATASETS["DDSM"]["NUMBER_OF_CLASSES"] = len(DATASETS["DDSM"]["CLASSES"])
