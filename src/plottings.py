import matplotlib.pyplot as plt


def scatterplot(x_coords, y_coords, color, save=False, name="test.png"):
    fig = plt.figure()
    plt.scatter(x_coords, y_coords, c=color)
    if save:
        fig.savefig(name)
    return fig


def main():
    pass


if __name__ == "__main__":
    main()
