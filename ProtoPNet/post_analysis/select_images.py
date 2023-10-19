import os

import pandas as pd

'''
    Usage: python3 select_images.py
    Saves sample_size number of examples each of the following scenarios:
        1. images classified correct by both models
        2. images misclassified by both models
        3. images classified correct by the first model but misclassified by the second one
        4. the invers case when the first model misclassifies and the second model makes correct prediction
'''

sample_size = 20

path = os.path.join(os.getcwd(), "saved_models", "resnet18")
model1 = os.path.join(path, "BOSCH_211", "8_9push0.8089")
model2 = os.path.join(path, "BOSCH_212_maxloss", "8nopush0.7803")
df1 = pd.read_csv(os.path.join(model1, "results.csv"), header=0)
df2 = pd.read_csv(os.path.join(model2, "results.csv"), header=0)
for d in [df1, df2]:
    d.drop(["act1", "act2", "act3"], axis=1, inplace=True)
df2.drop(["true_class"], axis=1, inplace=True)


df = df1.join(
    df2.set_index("image_path"),
    on="image_path",
    lsuffix="_avg",
    rsuffix="_max",
)

df_correct = df[
    (df["true_class"] == df["predicted_class_avg"])
    & (df["true_class"] == df["predicted_class_max"])
]
df_misclassified = df[
    (df["true_class"] != df["predicted_class_avg"])
    & (df["true_class"] != df["predicted_class_max"])
]

df1_correct = df[
    (df["true_class"] == df["predicted_class_avg"])
    & (df["true_class"] != df["predicted_class_max"])
]
df1_misclassified = df[
    (df["true_class"] != df["predicted_class_avg"])
    & (df["true_class"] == df["predicted_class_max"])
]

df_correct = df_correct.sample(sample_size)
df_misclassified = df_misclassified.sample(sample_size)
df1_correct = df1_correct.sample(sample_size)
df1_misclassified = df1_misclassified.sample(sample_size)

df_save = pd.concat(
    [df_correct, df_misclassified, df1_correct, df1_misclassified]
)

df_save.to_csv(os.path.join(path, "examples.csv"), index=False)
