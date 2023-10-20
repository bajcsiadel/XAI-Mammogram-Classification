import os

from PIL import Image
from ProtoPNet.config.settings import (
    base_architecture,
    experiment_run,
    img_shape,
    num_prototypes_per_class,
    train_dir,
)

categories = next(os.walk(train_dir))[1]
categories.sort()
prototype_images_dir = os.path.join(
    os.getcwd(), "saved_models", base_architecture, experiment_run, "img"
)
epochs = [20]

for k, category in enumerate(categories):
    collage = Image.new(
        "RGBA",
        (num_prototypes_per_class * img_shape[0], len(epochs) * img_shape[1]),
    )
    for i, epoch in enumerate(epochs):
        img_dir = os.path.join(prototype_images_dir, f"epoch-{epoch}")
        for j in range(num_prototypes_per_class):
            img_path = os.path.join(
                img_dir,
                (
                    "prototype-img-original_with_self_act"
                    f"{k*num_prototypes_per_class+j}.png"
                ),
            )
            img = Image.open(img_path).resize(img_shape)
            collage.paste(img, (j * img_shape[1], i * img_shape[0]))
    img_path = os.path.join(
        prototype_images_dir, f"{category}-prototype_evolution.png"
    )
    collage.save(img_path)
