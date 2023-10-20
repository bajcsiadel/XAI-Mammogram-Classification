import argparse
import os
import re

import pandas as pd
from ProtoPNet.config.settings import base_architecture, experiment_run

parser = argparse.ArgumentParser()
parser.add_argument("-f1", action="store_true")

args = parser.parse_args()
with_f1_score = args.f1

logfile_folder = os.path.join(
    os.getcwd(), "saved_models", base_architecture, experiment_run
)
logfile_path = os.path.join(logfile_folder, "train.log")
csv_path = os.path.join(logfile_folder, "loss_terms_and_accu.csv")

epoch_pattern = re.compile(r"^epoch:\s+([0-9]+)$")
train_pattern = re.compile(r"train$")
validation_pattern = re.compile(r"test$")
cross_entropy_pattern = re.compile(r"cross ent:\s+([.0-9e\-]+)$")
clustering_loss_pattern = re.compile(r"cluster:\s+([.0-9e\-]+)$")
separation_loss_pattern = re.compile(r"separation:\s+([.0-9e\-]+)$")
accuracy_pattern = re.compile(r"accu:\s+([.0-9e\-]+)%$")
micro_f1_pattern = re.compile(r"micro f1:\s+([.0-9e\-]+)%$")
macro_f1_pattern = re.compile(r"macro f1:\s+([.0-9e\-]+)%$")

train_columns = [
    "train_cross_entropy",
    "train_cluster",
    "train_separation",
    "train_accuracy",
]
validation_columns = [
    "validation_cross_entropy",
    "validation_cluster",
    "validation_separation",
    "validation_accuracy",
]
patterns = [
    cross_entropy_pattern,
    clustering_loss_pattern,
    separation_loss_pattern,
    accuracy_pattern,
]
if with_f1_score:
    train_columns += [
        "train_micro_f1",
        "train_macro_f1",
    ]
    validation_columns += [
        "validation_micro_f1",
        "validation_macro_f1",
    ]
    patterns += [
        micro_f1_pattern,
        macro_f1_pattern,
    ]


def search_pattern(logfile, pattern):
    while True:
        line = logfile.readline().strip()
        if not line:
            return False
        match = pattern.match(line)
        if match:
            return match


if __name__ == "__main__":
    df = pd.DataFrame(columns=train_columns + validation_columns)
    with open(logfile_path, "r") as logfile:
        last_epoch = -1
        next_epoch_match = search_pattern(logfile, epoch_pattern)
        while next_epoch_match:
            found_next_epoch = int(next_epoch_match.group(1))
            assert found_next_epoch == last_epoch + 1
            last_epoch = found_next_epoch

            row = {
                "epoch": last_epoch,
            }

            match = search_pattern(logfile, train_pattern)
            if not match:
                break
            for column_name, pattern in zip(train_columns, patterns):
                match = search_pattern(logfile, pattern)
                if not match:
                    break
                row[column_name] = float(match.group(1))

            search_pattern(logfile, validation_pattern)
            if not match:
                break
            for column_name, pattern in zip(validation_columns, patterns):
                match = search_pattern(logfile, pattern)
                if not match:
                    break
                row[column_name] = float(match.group(1))

            df.loc[last_epoch] = pd.Series(row)
            next_epoch_match = search_pattern(logfile, epoch_pattern)
    df.to_csv(csv_path, index=False)
