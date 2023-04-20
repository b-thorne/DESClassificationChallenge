import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle 

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
