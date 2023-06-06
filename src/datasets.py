import numpy as np
from typing import Tuple
from sklearn import datasets
from plottings import scatterplot
from sklearn import preprocessing
import torch


class Dataset:
    """
    A class to represent a dataset.

    ...

    Attributes:
    ----------
    X: np.ndarray
    y: np.ndarray
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, name="", homology=None):
        self.X = X
        self.y = y
        self.name = name
        self.size = self.X.shape[0]
        self.dim = self.X.shape[1]
        self.homology = homology

    def train_test_split(
        self, test_ratio=0.2, val=False, val_ratio=0.2, datatype="numpy", device=None
    ):
        data = np.hstack([self.X, self.y.reshape(-1, 1)])
        np.random.shuffle(data)
        if datatype == "torch":
            data = torch.Tensor(data)
            if device:
                data = data.to(device)
        test_thres = int(self.size * test_ratio)
        test_x, test_y = data[test_thres:, :-1], data[test_thres:, -1]
        train_x, train_y = data[:-test_thres, :-1], data[:-test_thres, -1]

        if val:
            val_thres = int(self.size * val_ratio)
            val_x, val_y = train_x[val_thres:, :], train_y[val_thres:, :]
            train_x, train_y = train_x[:-val_thres, :], train_y[:-val_thres, :]
            return train_x, train_y, val_x, val_y, test_x, test_y

        return train_x, train_y, test_x, test_y

    def plot_data(self, save=False):
        if self.dim == 2:
            return scatterplot(
                x_coords=self.X[:, 0], y_coords=self.X[:, 1], color=self.y, save=save
            )
        if self.dim == 3:
            return scatterplot(
                x_coords=self.X[:, 0],
                y_coords=self.X[:, 1],
                z_coords=self.X[:, 2],
                dim=3,
                color=self.y,
                save=save,
            )


class Circles(Dataset):
    def __init__(self, n_samples=2000, n_circles_per_class=4, margin=3, noise=0.15):
        self.n_samples = n_samples
        self.n_circles_per_class = n_circles_per_class
        homology = {
            0: [self.n_circles_per_class, self.n_circles_per_class],
            1: [self.n_circles_per_class, self.n_circles_per_class],
        }
        super().__init__(
            *self.__generate_data(margin, noise), name="circles", homology=homology
        )

    def __generate_data(self, margin, noise):
        circles_in_a_row = int(np.sqrt(self.n_circles_per_class))
        data_x, data_y = datasets.make_circles(n_samples=self.n_samples)
        for i in range(1, self.n_circles_per_class):
            new_data_x, new_data_y = datasets.make_circles(n_samples=self.n_samples)
            margin_0, margin_1 = (
                i // circles_in_a_row + noise,
                i % circles_in_a_row + noise,
            )
            new_data_x[:, 0], new_data_x[:, 1] = (
                new_data_x[:, 0] + margin * margin_0,  # + np.random.normal(0, noise),
                new_data_x[:, 1] + margin * margin_1,  # + np.random.normal(0, noise),
            )
            data_x = np.vstack([data_x, new_data_x])
            data_y = np.hstack([data_y, new_data_y])
        return data_x, data_y


class Tori(Dataset):
    def __init__(
        self,
        samples=150,
        shapeParam1=15,
        shapeParam2=2.5,
        shapeParam3=2.2,
        radius=1,
        rng=0.5,
        visual=False,
    ):
        self.samples = samples
        self.shapeParam1 = shapeParam1
        self.shapeParam2 = shapeParam2
        self.shapeParam3 = shapeParam3
        self.radius = radius
        self.visual = visual
        self.range = rng
        homology = {0: [8, 8], 0: [8, 8]}
        super().__init__(*self.__generate_data(), name="tori", homology=homology)

    def __draw_circle(self, r, center, n, rand=True):
        angles = np.linspace(start=0, stop=n, num=n) * (np.pi * 2) / n
        X = np.zeros(shape=(n, 2))
        X[:, 0] = np.sin(angles) * r
        X[:, 1] = np.cos(angles) * r

        if rand:
            return X + center + np.random.rand(n, 2) * r / self.shapeParam1
        else:
            return X + center

    def __gen_ring(self, center, flip, q=1.4, r=1):
        N_SAMPLES = self.samples
        X = np.zeros(shape=(2 * N_SAMPLES, 3))
        y = np.zeros(shape=(2 * N_SAMPLES,))

        X1 = self.__draw_circle(r=r, center=np.array((0, 0)), n=N_SAMPLES, rand=False)
        X2 = self.__draw_circle(r=r, center=np.array((0, 0)), n=N_SAMPLES, rand=False)

        X[0:N_SAMPLES, 0] = (X1[:, 0]) * self.shapeParam2 + np.random.uniform(
            low=-self.range, high=self.range, size=X1.shape[0]
        ) * q
        X[0:N_SAMPLES, 1] = (X1[:, 1]) * self.shapeParam2 + np.random.uniform(
            low=-self.range, high=self.range, size=X1.shape[0]
        ) * q
        X[0:N_SAMPLES, 2] = (
            np.random.uniform(low=-self.range, high=self.range, size=X1.shape[0]) * q
        )

        X[N_SAMPLES : 2 * N_SAMPLES, 0] = (
            X2[:, 0] * self.shapeParam3
            + np.random.uniform(low=-self.range, high=self.range, size=X1.shape[0]) * q
        )
        X[N_SAMPLES : 2 * N_SAMPLES, 1] = (
            X2[:, 1] * self.shapeParam3
            + np.random.uniform(low=-self.range, high=self.range, size=X1.shape[0]) * q
        )
        X[N_SAMPLES : 2 * N_SAMPLES, 2] = (
            np.random.uniform(low=-self.range, high=self.range, size=X1.shape[0]) * q
        )

        y[:] = flip
        y[0:N_SAMPLES] = flip

        X_total = X.copy() + np.array((self.shapeParam3, 0, 0))
        y_total = y.copy()

        X = np.zeros(shape=(2 * N_SAMPLES, 3))
        y = np.zeros(shape=(2 * N_SAMPLES,))

        X1 = self.__draw_circle(r=r, center=np.array((0, 0)), n=N_SAMPLES, rand=False)
        X2 = self.__draw_circle(r=r, center=np.array((0, 0)), n=N_SAMPLES, rand=False)

        X[0:N_SAMPLES, 0] = (X1[:, 0]) * self.shapeParam2 + np.random.uniform(
            low=-self.range, high=self.range, size=X1.shape[0]
        ) * q
        X[0:N_SAMPLES, 2] = (X1[:, 1]) * self.shapeParam2 + np.random.uniform(
            low=-self.range, high=self.range, size=X1.shape[0]
        ) * q
        X[0:N_SAMPLES, 1] = (
            np.random.uniform(low=-self.range, high=self.range, size=X1.shape[0]) * q
        )

        X[N_SAMPLES : 2 * N_SAMPLES, 0] = (
            X2[:, 0] * self.shapeParam3
            + np.random.uniform(low=-self.range, high=self.range, size=X1.shape[0]) * q
        )
        X[N_SAMPLES : 2 * N_SAMPLES, 2] = (
            X2[:, 1] * self.shapeParam3
            + np.random.uniform(low=-self.range, high=self.range, size=X1.shape[0]) * q
        )
        X[N_SAMPLES : 2 * N_SAMPLES, 1] = (
            np.random.uniform(low=-self.range, high=self.range, size=X1.shape[0]) * q
        )

        y[:] = 1 - flip
        y[0:N_SAMPLES] = 1 - flip

        X_total = np.concatenate((X_total, X), axis=0) + center
        y_total = np.concatenate((y_total, y), axis=0)

        return X_total, y_total

    def __generate_data(self, q=3):
        X1, y1 = self.__gen_ring((q, q, q), 0)
        X2, y2 = self.__gen_ring((-q, -q, q), 1)
        X3, y3 = self.__gen_ring((-q, q, -q), 0)
        X4, y4 = self.__gen_ring((q, -q, -q), 1)
        X5, y5 = self.__gen_ring((0, 0, 0), 0)
        X6, y6 = self.__gen_ring((-q, -q, -q), 0)
        X7, y7 = self.__gen_ring((q, q, -q), 1)
        X8, y8 = self.__gen_ring((-q, q, q), 0)
        X9, y9 = self.__gen_ring((q, -q, q), 1)

        X_total = np.concatenate((X1, X2), axis=0)
        y_total = np.concatenate((y1, y2), axis=0)

        X_total = np.concatenate((X_total, X3), axis=0)
        y_total = np.concatenate((y_total, y3), axis=0)

        X_total = np.concatenate((X_total, X4), axis=0)
        y_total = np.concatenate((y_total, y4), axis=0)

        X_total = np.concatenate((X_total, X5), axis=0)
        y_total = np.concatenate((y_total, y5), axis=0)

        X_total = np.concatenate((X_total, X6), axis=0)
        y_total = np.concatenate((y_total, y6), axis=0)

        X_total = np.concatenate((X_total, X7), axis=0)
        y_total = np.concatenate((y_total, y7), axis=0)

        X_total = np.concatenate((X_total, X8), axis=0)
        y_total = np.concatenate((y_total, y8), axis=0)

        X_total = np.concatenate((X_total, X9), axis=0)
        y_total = np.concatenate((y_total, y9), axis=0)

        X = X_total.copy()
        y = y_total.copy()

        max_abs_scaler = preprocessing.MaxAbsScaler()
        X = max_abs_scaler.fit_transform(X)

        return X, y


class Disks(Dataset):
    def __init__(
        self, random=False, grid_min=-9, grid_max=9, res=0.19, big_r=7, small_r=2.1, n=9
    ):
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.res = res
        self.random = random
        self.big_r = big_r
        self.small_r = small_r
        self.n = n
        homology = {0: [9, 0], 1: [1, 9]}
        super().__init__(*self.__generate_data(), name="disks", homology=homology)

    def __gen_grid(
        self,
        centers=[(0, 0), (8, 8)],
        radius=0.5,
        margin=0.5,
    ):
        x = np.arange(self.grid_min, self.grid_max, self.res)
        y = np.arange(self.grid_min, self.grid_max, self.res)
        xx, yy = np.meshgrid(x, y)
        if self.random:
            xx = xx + np.random.randn(xx.shape[0], xx.shape[1]) * 0.02
            yy = yy + np.random.randn(yy.shape[0], yy.shape[1]) * 0.02
        grid = np.dstack((xx, yy)).reshape(-1, 2)

        y = np.ones(shape=(len(grid), 2)) * 1

        for center in centers:
            for i, point in enumerate(grid):
                if np.linalg.norm(center - point) < radius + margin:
                    y[i, 0] = 3
                    y[i, 1] = 3

                if np.linalg.norm(center - point) < radius:
                    y[i, 0] = 0 * y[i, 0]
                    y[i, 1] = not y[i, 0]

            y = np.array(y)

            mask = y == 3
            label = y[:, 0]
            label = np.delete(label, np.where(mask[:, 0]))
            xx = list(grid[:, 0])
            yy = list(grid[:, 1])

            xx = np.delete(xx, np.where(mask[:, 0]))
            yy = np.delete(yy, np.where(mask[:, 0]))
            grid = np.ones(shape=(len(xx), 2))

            grid[:, 0] = xx
            grid[:, 1] = yy

            y = np.ones(shape=(len(xx), 2))
            y[:, 0] = label
            y[:, 1] = 1 - label

        return (grid, y[:, 0])

    def __generate_data(self):
        centers = [
            (
                np.cos(2 * np.pi / (self.n - 1) * x) * self.big_r,
                np.sin(2 * np.pi / (self.n - 1) * x) * self.big_r,
            )
            for x in range(0, self.n)
        ] + [0, 0]
        X, y = self.__gen_grid(
            centers=centers,
            radius=self.small_r,
        )

        return X, y


def main():
    tori = Tori()
    print(tori.name, tori.size, tori.y[tori.y == 0].shape, tori.y[tori.y == 1].shape)
    tori.plot_data(save=True)


if __name__ == "__main__":
    main()
