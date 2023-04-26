import matplotlib.pyplot as plt


def scatterplot(
    x_coords,
    y_coords,
    color,
    z_coords=None,
    dim=2,
    engine="matplotlib",
    save=False,
    name="test.png",
):
    if engine == "matplotlib":
        if dim == 2:
            fig = plt.figure()
            ax = fig.add_subplot(projection="2d")
            ax.scatter(x_coords, y_coords, c=color)
        if dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.scatter(x_coords, y_coords, z_coords, c=color)
    if save:
        fig.savefig(name)
    return fig


def plot_lines(
    x_ranges,
    y_ranges,
    stds=None,
    labels=None,
    titles=None,
    fig_title=None,
    xlabels=None,
    share_x_range=True,
    save=True,
    filename="",
    **kwargs
):
    fig, axs = plt.subplots(**kwargs)
    axs = axs.flatten()
    n_axes = len(y_ranges)
    if share_x_range:
        x_ranges = [x_ranges] * n_axes
    for i, x_range in enumerate(x_ranges):
        for j, y_range in enumerate(y_ranges[i]):
            axs[i].plot(x_range, y_range, label=labels[i][j])
            axs[i].fill_between(
                x_range, y_range - stds[i][j], y_range + stds[i][j], alpha=0.1
            )
        axs[i].set_xlabel(xlabels[i])
        axs[i].set_title(titles[i])
        axs[i].legend(loc="best")
    fig.suptitle(fig_title)
    if save:
        fig.savefig(filename)


def main():
    pass


if __name__ == "__main__":
    main()
