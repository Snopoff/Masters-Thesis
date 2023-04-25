from sklearn import datasets

from utils.plottings import scatterplot


def generate_circles(n_samples=2000):
    circles = datasets.make_circles(n_samples=n_samples)
    return circles


def main():
    circles = generate_circles()
    fig = scatterplot(circles[0][:, 0], circles[0][:, 1], color=circles[1])
    fig.savefig("test.png")
    fig.show()


if __name__ == "__main__":
    main()
