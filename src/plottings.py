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
    x_ranges, y_ranges, stds=None, labels=None, title=None, share_x_range=True, **kwargs
):
    fig, axs = plt.subplots(kwargs)
    axs = axs.flatten()


def main():
    pass


if __name__ == "__main__":
    main()
