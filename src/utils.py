import numpy as np


def train_test_split(X, y, val=False, test_ratio=0.2, val_ratio=0.2):
    size = y.shape[0]

    data = np.hstack([X[0], y[1].reshape(-1, 1)])
    np.random.shuffle(data)

    if val:
        thres_test = int(test_ratio * size)
        thres_val = int(val_ratio * (size - thres_test))
        train_x, train_y = data[:-thres_test, :-1], data[:-thres_test, -1]
        test_x, test_y = data[thres_test:, :-1], data[thres_test:, -1]

    else:
        thres_test = int(test_ratio * size)
        train_x, train_y = data[:-thres_test, :-1], data[:-thres_test, -1]
        test_x, test_y = data[thres_test:, :-1], data[thres_test:, -1]

        return train_x, test_x, train_y, test_y
