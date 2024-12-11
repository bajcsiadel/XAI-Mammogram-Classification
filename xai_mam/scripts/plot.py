import argparse
import os
import pathlib
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from icecream import ic

parser = argparse.ArgumentParser(
    "plot.py",
    description="Plot the train and test accuracies during "
    "the training process of the ProtoPNet model",
)
parser.add_argument(
    "--output-file",
    dest="output_file",
    type=pathlib.Path,
    help="Path to the output file recorded during training. "
    "Specify absolute path or relative to `runs/` directory",
)
parser.add_argument(
    "-s",
    "--save",
    action="store_true",
    help="Save the generated plot near the specified output file",
)
parser.add_argument(
    "--show-guide-lines",
    dest="show_guide_lines",
    action="store_true",
    help="Show guide lines for warm and push epochs",
)


load_dotenv()
assert os.getenv("RUNS_DIR") is not None, "PROJECT_ROOT is not set in .env"

args = parser.parse_args()
output_file = args.output_file
if output_file.root == "":
    # if the path is relative then concatenate it to the project root
    output_file = pathlib.Path(os.getenv("RUNS_DIR")) / output_file

if not output_file.exists():
    raise ValueError(f"Output file {output_file} does not exist")

if output_file.is_dir():
    if not (output_file := output_file / "train_model.csv").exists():
        raise ValueError(
            f"`train_model.csv` does not exist in the "
            f"given output directory {output_file}"
        )
else:
    if output_file.suffix != ".csv":
        raise ValueError(
            f"Output file {output_file} is not a csv file. Extension not supported!"
        )

train_data = pd.read_csv(output_file, header=0)
print(f"Used output file: {output_file}")

warm_epochs = (
    train_data["phase"].apply(lambda value: value.startswith("warm")).sum() // 2
)
push_epochs = train_data[
    train_data["phase"].apply(lambda value: value.startswith("push")).values
]["epoch"].values

train_data = train_data[
    train_data["phase"]
    .apply(lambda value: re.fullmatch(r"(warm )?(train|validation)", value) is not None)
    .values
]
train_data["phase"] = train_data["phase"].apply(lambda value: value.split(" ")[-1])

ax = sns.lineplot(
    train_data, x="epoch", y="accuracy", hue="phase"
)

if "only" not in str(output_file) and args.show_guide_lines:
    print(f"Number of warm epochs: {warm_epochs}")
    print(f"Push epochs: {push_epochs}")

    ax.axvline(
        x=warm_epochs,
        color=sns.set_hls_values("black", l=0.5),
        linestyle="--",
        label="end of warm",
    )

    line = None
    for push_epoch in push_epochs:
        line = ax.axvline(
            x=push_epoch, color=sns.set_hls_values("orange", l=0.8), linestyle="--"
        )
    # set label to only the last line
    if line is not None:
        line.set_label("push")

# show legend with the updated labels (defined by the axvline)
plt.tight_layout()

if args.save:
    plt.savefig(
        output_file.with_name("plot_train_test_accuracies.png"),
        dpi=300,
        bbox_inches="tight",
    )
else:
    plt.show()
