from src.datasets import Circles, Tori
from src.models import Classifier1L, ClassifierAL
from src.experiments import ActivationExperiments
import argparse
import time

DATASETS_DICT = {
    "circles": Circles,
    "tori": Tori,
}

EXPERIMENTS_DICT = {
    "activation": ActivationExperiments,
}

MODELS_DICT = {"1L": Classifier1L, "AL": ClassifierAL}


def main():
    parser = argparse.ArgumentParser(
        description="Run the experiments for the Masters thesis."
    )
    parser.add_argument(
        "--e",
        type=str,
        help="which experiment to choose",
        default="activation",
        choices=["activation"],
    )
    parser.add_argument(
        "--l", type=int, default=1, help="num of layers to be used in a model"
    )
    parser.add_argument(
        "--d",
        type=int,
        default=1,
        help="dimension of hidden layers to be used in a model",
    )
    parser.add_argument(
        "--data",
        type=str,
        action="append",
        default=["circles", "tori"],
        choices=["circles", "tori"],
        help="datasets names to be used in the experiment",
    )
    parser.add_argument(
        "--act",
        type=str,
        action="append",
        default=["split_tanh", "split_sincos", "relu"],
        choices=["split_tanh", "split_sign", "split_sincos", "relu"],
        help="activation functions names to be used in the experiment",
    )
    parser.add_argument(
        "--ne",
        type=int,
        default=2,
        help="number of experiments to play for each model",
    )
    parser.add_argument(
        "--m",
        type=str,
        default="AL",
        choices=["1L", "AL"],
        help="Which model to choose in the experiments",
    )

    args = parser.parse_args()

    datasets = [DATASETS_DICT[dataset]() for dataset in args.data]
    experiment = EXPERIMENTS_DICT[args.e](
        model=MODELS_DICT[args.m],
        datasets=datasets,
        n_experiments=args.ne,
        num_of_hidden_layers=range(1, args.l + 1),
        dim_of_hidden_layers=(3, args.d + 3),
        list_of_activations=args.act,
        verbose=False,
    )
    start_time = time.time()
    experiment.run_experiments()
    end_time = time.time()
    print("Time spent = {:.2f} min".format((end_time - start_time) / 60))


if __name__ == "__main__":
    main()
