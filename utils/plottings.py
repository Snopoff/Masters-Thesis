import matplotlib.pyplot as plt


def scatterplot(x_coords, y_coords, color):
    fig = plt.figure()
    plt.scatter(x_coords, y_coords, c=color)
    return fig


def main():
    pass


if __name__ == "__main__":
    main()
