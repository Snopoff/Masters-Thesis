from ripser import ripser
from persim import plot_diagrams
import numpy as np
import scipy.sparse as ss

# from .datasets import Circles
from .plottings import plot_dgm
from .utils import obtain_points_for_each_label


def compute_homology(data, maxdim=2, subsample_size=1000, **kwargs):
    return ripser(data, maxdim=maxdim, n_perm=subsample_size, **kwargs)


def topological_complexity(
    X, labels=None, maxdim=1, subsample_size=1000, drop_zeroth=True, scale=None
):
    if labels is not None:
        data = obtain_points_for_each_label(X, labels)
    else:
        data = {-1: X}
    b_numbers = dict()
    for label, X in data.items():
        res = compute_homology(X, maxdim, subsample_size)["dgms"]
        if drop_zeroth:
            res = res[1:]
        if scale is not None:
            if isinstance(scale, int):
                res_at_scale = [
                    gen
                    for hom in res
                    for gen in hom
                    if gen[0] <= scale and scale < gen[1]
                ]
                b_numbers[label] = len(res_at_scale)
            if isinstance(scale, list):
                b_numbers[label] = dict()
                for s in scale:
                    res_at_scale = [
                        gen for hom in res for gen in hom if gen[0] <= s and s < gen[1]
                    ]
                    b_numbers[label][s] = len(res_at_scale)
        else:
            b_numbers[label] = sum([hom.shape[0] for hom in res])
    return b_numbers


def create_sublevelset_filtration(x):
    N = x.shape[0]
    I = np.arange(N - 1)
    J = np.arange(1, N)
    V = np.maximum(x[:-1], x[1:])

    I = np.concatenate((I, np.arange(N)))
    J = np.concatenate((J, np.arange(N)))
    V = np.concatenate((V, x))
    D = ss.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()

    return D


def topo_simplification(ts, distance_to_diagonal=1, plot_diagram=False):
    x = ts.to_numpy()

    dist_matrix = create_sublevelset_filtration(x)
    homology = compute_homology(
        dist_matrix, maxdim=0, subsample_size=None, distance_matrix=True
    )
    dgm0 = homology["dgms"][0]
    points_to_return = dgm0[:, 1] - dgm0[:, 0] >= distance_to_diagonal
    dgm0_ret = dgm0[points_to_return, :]

    if plot_diagram:
        plot_diagrams([dgm0, dgm0_ret], labels=["$H_0$", "Simplified $H_0$"])

    ts_to_return = (
        [
            int(np.where(np.isclose(x, point))[0][0])
            for point in dgm0_ret.reshape(-1)[:-1]
        ]
        + [ts.shape[0] - 1]
        + [0]
    )
    ts_to_return = ts[ts_to_return].sort_index().drop_duplicates()
    return ts_to_return


def main():
    circle = Circles()
    scale = [0.5, 1, 1.5]
    res = topological_complexity(circle.X, labels=circle.y, scale=scale)
    print(res)


if __name__ == "__main__":
    main()
