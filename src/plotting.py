import matplotlib.pyplot as plt
from sklearn.metrics import det_curve


def plot_batch(batch, col_titles=["Template", "Search", "Diff"]):
    (inputs, labels) = batch
    nbatch = inputs.shape[0]
    ncols = 3
    figsize = (3 * 2, nbatch * 2)
    fig, axes = plt.subplots(nrows=nbatch, ncols=ncols, figsize=figsize)
    fig.subplots_adjust(wspace=0.05)
    for row in range(nbatch):
        for col in range(ncols):
            data = inputs[row, col, :, :]
            ax = axes[row, col]

            ax.imshow(data)

            c = "green" if labels[row] else "red"
            for side in ["bottom", "top", "left", "right"]:
                ax.spines[side].set_color(c)
                ax.spines[side].set_linewidth(2)

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    fig.tight_layout()

    for (col, title) in enumerate(col_titles):
        axes[0, col].set_title(title)

    return fig


def plot_mdr(y_true, y_score,
             anchor_points=[[0.03, 0.04, 0.05], [0.037, 0.024, 0.015]]):
    fpr, fnr, _ = det_curve(y_true, y_score)
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel("MDR")
    ax.set_ylabel("FPR")
    ax.plot(fpr, fnr, label="CNN")
    ax.set_xlim(0, 0.1)
    ax.set_ylim(0, 0.05)
    ax.scatter(anchor_points[0], anchor_points[1], marker="s", color="k",
               label="Goldstein 2015")
    ax.legend(loc=3, frameon=False)
    return fig
