from datasets import Dataset, Circles, Tori
from model import Classifier1L
from train import train_eval_loop
from plottings import plot_lines
import numpy as np
import random
import torch


def set_state(random_state):
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)


def run_experiment(
    dataset: Dataset,
    random_state: int,
    num_of_hidden: int,
    dim_of_hidden: int,
    activation: str,
    epochs: int,
    test_ratio=0.2,
    verbose=False,
):
    set_state(random_state)
    model = Classifier1L(
        dim_of_in=dataset.dim,
        num_of_hidden=num_of_hidden,
        dim_of_hidden=dim_of_hidden,
        activation=activation,
    )

    if verbose:
        print(
            "Dataset is {}\tLayers signature is {}".format(
                dataset.name, (num_of_hidden, dim_of_hidden, activation)
            )
        )

    return train_eval_loop(
        model,
        dataset,
        epochs=epochs,
        test_ratio=test_ratio,
        return_losses=True,
    )


def run_experiments_for_given_model(
    dataset,
    n_experiments,
    num_of_hidden,
    dim_of_hidden,
    activation,
    epochs,
    verbose=False,
):
    train_loss, test_loss = [], []
    random_states = range(1, n_experiments + 1)
    for random_state in random_states:
        train_losses, test_losses = run_experiment(
            dataset,
            random_state,
            num_of_hidden,
            dim_of_hidden,
            activation,
            epochs=epochs,
            verbose=verbose,
        )
        if verbose:
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


def plot_results(
    train_info,
    test_info,
    datasets_names,
    dim_of_hidden_layers,
    num_of_hidden_layers,
    epochs,
):
    x_range = range(epochs)
    num_dims = len(dim_of_hidden_layers)
    xlabels = ["epochs"] * num_dims
    num_layers = len(num_of_hidden_layers)
    for key, value in enumerate(datasets_names):
        data_train = [
            train_info[i] for i in range(len(train_info)) if train_info[i][0][0] == key
        ]
        for hid_layer in num_of_hidden_layers:
            data_for_given_layer = [
                data_train[i]
                for i in range(len(data_train))
                if data_train[i][0][1] == hid_layer
            ]
            y_ranges = [None] * num_dims
            stds = [None] * num_dims
            labels = [None] * num_dims
            titles = [None] * num_dims
            for i, dim in enumerate(dim_of_hidden_layers):
                data_for_given_dim = [
                    data_for_given_layer[i]
                    for i in range(len(data_for_given_layer))
                    if data_for_given_layer[i][0][2] == dim
                ]
                y_ranges[i] = [data[1] for data in data_for_given_dim]
                labels[i] = [
                    "train loss w/" + data[0][-1] for data in data_for_given_dim
                ]
                stds[i] = [data[2] for data in data_for_given_dim]
                titles[i] = "Dimension of hidden layers={}".format(dim)
            fig_title = "Dataset {}; № of layers = {}".format(value, hid_layer)
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
                filename=fig_title,
                ncols=4,
                nrows=1 + num_layers // 4,
                figsize=(15, 10),
            )


def run_experiments(
    datasets,
    n_experiments=2,
    num_of_hidden_layers=range(1, 5, 2),
    dim_of_hidden_layers=range(3, 4),
    list_of_activations=[
        "split_tanh",
        "split_sign",
        "relu",
    ],  # ["split_tanh", "split_sign", "split_sincos", "relu"],
    epochs=500,
    verbose=True,
):
    train_info, test_info = [], []
    for i, dataset in enumerate(datasets):
        print("Experimenting with dataset {}".format(dataset.name))
        for num_of_hidden in num_of_hidden_layers:
            for dim_of_hidden in dim_of_hidden_layers:
                for activation in list_of_activations:
                    (
                        train_loss_mean,
                        test_loss_mean,
                        train_loss_std,
                        test_loss_std,
                    ) = run_experiments_for_given_model(
                        dataset,
                        n_experiments,
                        num_of_hidden,
                        dim_of_hidden,
                        activation,
                        epochs,
                        verbose,
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

    datasets_names = [dataset.name for dataset in datasets]
    print("Plotting the results ...")
    plot_results(
        train_info=train_info,
        test_info=test_info,
        datasets_names=datasets_names,
        dim_of_hidden_layers=dim_of_hidden_layers,
        num_of_hidden_layers=num_of_hidden_layers,
        epochs=epochs,
    )


if __name__ == "__main__":
    circles, tori = Circles(), Tori()
    datasets = [circles, tori]
    run_experiments(datasets)
