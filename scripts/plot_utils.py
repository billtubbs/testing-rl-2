import numpy as np
import matplotlib.pyplot as plt


def remove_outliers(data, conf_int=0.98):
    q_low = (1.0 - conf_int) / 2
    y_low = data.quantile(q_low)
    y_hi = data.quantile(1.0 - q_low)
    return data.loc[(data < y_hi) & (data > y_low)]


def choose_bin_width(y_span, n_bins_min=20):
    """Selects a bin width to divide up the span between y_min and y_max
    that is a sensible decimal number such as 0.1, 0.2, 0.5, 1, 2, 5, ...etc.

    Example:
    >>> choose_bin_width(5)
    0.2
    >>> choose_bin_width(100)
    5.0
    """
    bin_width = 10 ** np.floor(np.log10(y_span * 5 / n_bins_min))
    for m in [2.0, 2.5, 2.0, 2.0]:
        if (y_span / bin_width) >= n_bins_min:
            break
        bin_width = bin_width / m
    return bin_width


def make_histogram(
    data,
    ax=None,
    y_min=None,
    y_max=None,
    bin_width=None,
    n_bins_min=20,
    x_label=None,
    density=False,
    title=None,
    figsize=(7, 4.5),
    **kwargs,
):
    """Makes a histogram with automatically selected bin edges that line
    up with sensible decimal values like 0.1, 0.2, 0.5, 1, 2, 5, ...etc.
    """
    if y_min is None:
        y_min = data.min()
    if y_max is None:
        y_max = data.max()
    if bin_width is None:
        bin_width = choose_bin_width(y_max - y_min, n_bins_min=n_bins_min)
    y_min = np.floor(y_min / bin_width) * bin_width
    y_max = np.ceil(y_max / bin_width) * bin_width
    n_bins = int(np.round((y_max - y_min) / bin_width))
    bin_edges = np.linspace(
        y_min - bin_width / 2, y_max + bin_width / 2, n_bins + 2
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    data.hist(ax=ax, bins=bin_edges, density=density, **kwargs)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if density:
        ax.set_ylabel("Density")
    else:
        ax.set_ylabel("Count")
    if title:
        ax.set_title(title)
    return fig, ax
