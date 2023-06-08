from .datasets import Dataset, Circles, Tori, Disks
from .models import ClassifierAL
from .train import train_eval_loop
from .plottings import plot_lines, lineplot
from .utils import mkdir_p
from typing import List
import numpy as np
import random
import torch
import tqdm
import time

DIRNAME = "images/"


def set_state(random_state):
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)


class ActivationExperiments:
    def __init__(
        self,
        model,
        datasets: List,
        n_experiments=30,
        num_of_hidden_layers=range(1, 11),
        dim_of_hidden_layers=range(3, 11),
        list_of_activations=[
            "split_tanh",
            "split_sign",
            "split_sincos",
            "relu",
        ],
        test_ratio=0.2,
        epochs=5000,
        verbose=True,
    ):
        self.model = model
        self.datasets = datasets
        self.n_experiments = n_experiments
        self.num_of_hidden_layers = num_of_hidden_layers
        self.dim_of_hidden_layers = dim_of_hidden_layers
        self.list_of_activations = list_of_activations
        self.test_ratio = test_ratio
        self.epochs = epochs
        self.verbose = verbose

    def run_experiment(
        self,
        dataset: Dataset,
        random_state: int,
        num_of_hidden: int,
        dim_of_hidden: int,
        activation: str,
    ):
        set_state(random_state)
        model = self.model(
            dim_of_in=dataset.dim,
            num_of_hidden=num_of_hidden,
            dim_of_hidden=dim_of_hidden,
            activation=activation,
        )

        if self.verbose:
            print(
                "Dataset is {}\tLayers signature is {}".format(
                    dataset.name, (num_of_hidden, dim_of_hidden, activation)
                )
            )

        return train_eval_loop(
            model,
            dataset,
            epochs=self.epochs,
            test_ratio=self.test_ratio,
            return_losses=True,
        )

    def run_experiments_for_given_model(
        self, dataset, num_of_hidden, dim_of_hidden, activation
    ):
        train_loss, test_loss = [], []
        random_states = range(1, self.n_experiments + 1)
        for random_state in tqdm.tqdm(random_states, desc=" experiments", position=0):
            train_losses, test_losses = self.run_experiment(
                dataset, random_state, num_of_hidden, dim_of_hidden, activation
            )
            if self.verbose:
                print(
                    "Last training loss={}\tLast test loss={}".format(
                        train_losses[-1], test_losses[-1]
                    )
                )
            train_loss.append(train_losses)
            test_loss.append(test_losses)

        train_loss, test_loss = np.vstack(train_loss), np.vstack(test_loss)
        train_loss_mean, test_loss_mean = np.mean(train_loss, axis=0), np.mean(
            test_loss, axis=0
        )
        train_loss_std, test_loss_std = np.std(train_loss, axis=0), np.std(
            test_loss, axis=0
        )
        return train_loss_mean, test_loss_mean, train_loss_std, test_loss_std

    def plot_results(self, results, generic_label="train loss w/", dirname=DIRNAME):
        mkdir_p(dirname)
        datasets_names = [dataset.name for dataset in self.datasets]
        x_range = range(self.epochs)
        num_dims = len(self.dim_of_hidden_layers)
        xlabels = ["epochs"] * num_dims
        for key, value in enumerate(datasets_names):
            data_train = [
                results[i] for i in range(len(results)) if results[i][0][0] == key
            ]
            for hid_layer in self.num_of_hidden_layers:
                data_for_given_layer = [
                    data_train[i]
                    for i in range(len(data_train))
                    if data_train[i][0][1] == hid_layer
                ]
                y_ranges = [None] * num_dims
                stds = [None] * num_dims
                labels = [None] * num_dims
                titles = [None] * num_dims
                for i, dim in enumerate(self.dim_of_hidden_layers):
                    data_for_given_dim = [
                        data_for_given_layer[i]
                        for i in range(len(data_for_given_layer))
                        if data_for_given_layer[i][0][2] == dim
                    ]
                    y_ranges[i] = [data[1] for data in data_for_given_dim]
                    labels[i] = [
                        generic_label + data[0][-1] for data in data_for_given_dim
                    ]
                    stds[i] = [data[2] for data in data_for_given_dim]
                    titles[i] = "Dimension of hidden layers={}".format(dim)
                fig_title = "Dataset {}; № of layers = {}".format(value, hid_layer)
                filename = dirname + fig_title
                plot_lines(
                    x_range,
                    y_ranges,
                    stds=stds,
                    labels=labels,
                    titles=titles,
                    fig_title=fig_title,
                    xlabels=xlabels,
                    share_x_range=True,
                    save=True,
                    filename=filename,
                    ncols=4,
                    nrows=num_dims // 4,
                    figsize=(15, 10),
                )

    def save_results(self, results, generic_label, dirname):
        pass

    def run_experiments(self):
        train_info, test_info = [], []
        for i, dataset in enumerate(self.datasets):
            print("Experimenting with dataset {}".format(dataset.name))
            for num_of_hidden in self.num_of_hidden_layers:
                for dim_of_hidden in self.dim_of_hidden_layers:
                    for activation in self.list_of_activations:
                        print(
                            "№ of layers = {};\t dim of hidden = {};\t activation is {}".format(
                                num_of_hidden, dim_of_hidden, activation
                            )
                        )
                        (
                            train_loss_mean,
                            test_loss_mean,
                            train_loss_std,
                            test_loss_std,
                        ) = self.run_experiments_for_given_model(
                            dataset, num_of_hidden, dim_of_hidden, activation
                        )
                        train_info.append(
                            (
                                (i, num_of_hidden, dim_of_hidden, activation),
                                train_loss_mean,
                                train_loss_std,
                            )
                        )
                        test_info.append(
                            (
                                (i, num_of_hidden, dim_of_hidden, activation),
                                test_loss_mean,
                                test_loss_std,
                            )
                        )
        print("Plotting the results ...")
        self.plot_results(
            results=train_info,
            generic_label="train loss w/",
            dirname=DIRNAME + "activations/train/",
        )
        self.plot_results(
            results=test_info,
            generic_label="test loss w/",
            dirname=DIRNAME + "activations/test/",
        )


class TopologyChangeExperiments:
    """
    We firstly train a model (via train_eval_loop),
    then we pick val data and run it through the model again,
    but now we track the topology changes
    """

    def __init__(
        self,
        model,
        datasets: List,
        model_config={"num_of_hidden": 1, "dim_of_hidden": 3},
        n_experiments=30,
        list_of_activations=[
            "split_tanh",
            "split_sign",
            "split_sincos",
            "relu",
        ],
        test_ratio=0.2,
        epochs=5000,
    ) -> None:
        self.model = model
        self.datasets = datasets
        self.model_config = model_config
        self.n_experiments = n_experiments
        self.list_of_activations = list_of_activations
        self.test_ratio = test_ratio
        self.epochs = epochs

    def plot_results(self, results):
        plot_dir = DIRNAME + "topoChanges/"
        mkdir_p(plot_dir)
        for dataset, info in results.items():
            title = dataset.name
            plot_path = plot_dir + title
            labels = list(info.keys())
            values = list(info.values())
            x_range = range(1, values[0][0].shape[0] + 1)
            y_ranges, stds = list(zip(*values))
            lineplot(
                x_range,
                y_ranges,
                stds,
                title,
                labels,
                xlabel="layers",
                ylabel="TC",
                filename=plot_path,
            )

    def run_experiments(self, verbose=False, plot_results=True, homology_of_label=-1):
        results = {}
        for dataset in self.datasets:
            results[dataset] = {}
            if verbose:
                print("Working with dataset {}".format(dataset))
            for activation in self.list_of_activations:
                print("Working with activation {}".format(activation))
                topo_changes = np.zeros(
                    (self.n_experiments, self.model_config["num_of_hidden"] + 2)
                )
                for i in range(self.n_experiments):
                    model = self.model(
                        dim_of_in=dataset.dim,
                        num_of_hidden=self.model_config["num_of_hidden"],
                        dim_of_hidden=self.model_config["dim_of_hidden"],
                        activation=activation,
                    )
                    res = train_eval_loop(
                        model,
                        dataset,
                        epochs=self.epochs,
                        test_ratio=self.test_ratio,
                        return_topo_changes=True,
                    )
                    topo_changes[i] = [d[homology_of_label] for d in res]
                mean_topo_change = np.mean(topo_changes, axis=0)
                std_topo_change = np.mean(topo_changes, axis=0)
                results[dataset][activation] = (mean_topo_change, std_topo_change)
                print("Mean topology changes = {}".format(mean_topo_change))

        if plot_results:
            self.plot_results(results)

        return results


def main_activations():
    circles, tori, disks = Circles(), Tori(), Disks()
    datasets = [circles, tori, disks]
    experiment = ActivationExperiments(
        model=ClassifierAL,
        datasets=datasets,
        n_experiments=2,
        num_of_hidden_layers=range(1, 2),
        dim_of_hidden_layers=(3, 4),
        list_of_activations=["split_tanh", "split_sincos", "relu"],
        verbose=False,
    )
    start_time = time.time()
    experiment.run_experiments()
    end_time = time.time()
    print("Time spent = {:.2f} min".format((end_time - start_time) / 60))


def main_topo_changes():
    datasets = [Circles()]
    experiment = TopologyChangeExperiments(
        model=ClassifierAL,
        datasets=datasets,
        n_experiments=2,
        list_of_activations=["relu", "split_tanh", "split_sincos"],
        model_config={"num_of_hidden": 3, "dim_of_hidden": 5},
    )
    start_time = time.time()
    experiment.run_experiments()
    end_time = time.time()
    print("Time spent = {:.2f} min".format((end_time - start_time) / 60))


if __name__ == "__main__":
    main_topo_changes()
