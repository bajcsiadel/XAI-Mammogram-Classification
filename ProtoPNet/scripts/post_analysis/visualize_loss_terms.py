import os

import matplotlib.pyplot as plt
import pandas as pd
from extract_loss_terms_from_log import test_columns, train_columns
from ProtoPNet.config.settings import base_architecture, experiment_run
from extract_loss_terms_from_log import train_columns, validation_columns

log_folder = os.path.join(
    os.getcwd(), "saved_models", base_architecture, experiment_run
)
csv_path = os.path.join(log_folder, "loss_terms_and_accu.csv")
png_path = os.path.join(log_folder, "loss_terms_and_accu.png")

labels = ["CrsEnt", "Clus", "Sep"]

if __name__ == "__main__":
    df = pd.read_csv(csv_path)

    fig, (
        (ax_train_accu, ax_validation_accu),
        (ax_train_loss, ax_validation_loss),
    ) = plt.subplots(2, 2)
    ax_train_accu.sharey(ax_validation_accu)
    ax_train_loss.sharey(ax_validation_loss)
    ax_train_loss.set_yscale("log")

    ax_train_accu.set_title("Train")
    ax_validation_accu.set_title("Validation")
    ax_train_accu.set_ylabel("accuracy")
    ax_train_loss.set_ylabel("loss terms")
    ax_train_loss.set_xlabel("epochs")
    ax_validation_loss.set_xlabel("epochs")

    df.plot.line(y=train_columns[-1], legend=False, ax=ax_train_accu)
    df.plot.line(y=validation_columns[-1], legend=False, ax=ax_validation_accu)
    df.plot.line(y=train_columns[:-1], legend=False, ax=ax_train_loss)
    df.plot.line(y=validation_columns[:-1], legend=False, ax=ax_validation_loss)

    plt.legend(labels)

    plt.savefig(png_path, dpi=500)
