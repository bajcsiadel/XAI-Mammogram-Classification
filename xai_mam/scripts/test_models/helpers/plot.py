from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_plot_from_data_frame(
    data_frame: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    title: str = "",
    save_path: Path = None,
    **kwargs,
):
    """
    Create a plot from a pandas DataFrame. ``x``, ``y``, and ``hue`` must
    be column names from the given dataframe.

    :param data_frame: data
    :param x: column name for x-axis
    :param y: column name for y-axis
    :param hue: column name for hue
    :param title: title of the plot
    :param save_path: path to save the plot
    :param kwargs: additional parameters for the plot
    """
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(rc=custom_params)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=data_frame, x=x, y=y, hue=hue, ax=ax)

    # set labels and title if specified
    if title:
        ax.set_title(title)
    ax.set_xlabel(kwargs.get("x_label", x))
    ax.set_ylabel(kwargs.get("y_label", y))

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=kwargs.get("dpi", 300),
            bbox_inches=kwargs.get("bbox_inches", "tight"),
        )
    else:
        fig.show()
