from ripser import ripser
from persim import plot_diagrams
import numpy as np
import scipy.sparse as ss


def compute_homology(data, maxdim=2, subsample_size=None, **kwargs):
    return ripser(data, maxdim=maxdim, n_perm=subsample_size, **kwargs)


def topological_complexity(
    X, labels=None, maxdim=1, subsample_size=None, drop_zeroth=True
):
    if labels != None:
        label_vals = np.unique(labels)
        b_numbers = np.zeros_like(label_vals)
        for i, label in enumerate(label_vals):
            label_mask = labels == label
            X_label = X[label_mask, :]
            res = compute_homology(X_label, maxdim, subsample_size)["dgms"]
            if drop_zeroth:
                res = res[1:]
            b_numbers[i] = sum([hom.shape[0] for hom in res])
    else:
        res = compute_homology(X, maxdim, subsample_size)["dgms"]
        if drop_zeroth:
            res = res[1:]
        b_numbers = sum([hom.shape[0] for hom in res])

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
