import os
import pickle
from generate_train_test import get_dataset
from federated_learning.datasets.data_distribution import generate_iid_data
from loguru import logger
import argparse
from federated_learning.arguments import Arguments
import random

SEED = 1
random.seed(SEED)

if __name__ == "__main__":
    # Initialize logger
    handler = logger.add("logs/0_clients.log", enqueue=True)

    parser = argparse.ArgumentParser(description="A Clean-Label Attack in FL")
    parser.add_argument("--config", type=str, help="Configuration file", default="federated_learning/config/test.json")
    parser.add_argument("--data-distribution", type=str, help="iid or non-iid", default="iid")

    config = parser.parse_args().config
    absolute_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config)

    args = Arguments(logger, config_filepath=absolute_config_path)

    kwargs = {"num_workers": 0, "pin_memory": True} if args.cuda else {}
    data_distribution = parser.parse_args().data_distribution
    train_dataset, test_dataset = get_dataset(args, kwargs)
    if data_distribution == "iid":
        train_loaders, train_indices, test_data_loader = generate_iid_data(train_dataset, test_dataset, args, kwargs)
    elif data_distribution == "non-iid":
        pass

    # Save train loaders, test loader, train indices
    n_clients = args.args_dict.fl_training.num_clients
    train_loaders_path = f"./data_loaders/cifar10/iid/train_loaders_iid_n_clients_{n_clients}"
    test_data_loader_path = f"./data_loaders/cifar10/iid/test_data_loader_iid_n_clients_{n_clients}"
    train_indices_path = f"./data_loaders/cifar10/iid/train_indices_iid_n_clients_{n_clients}"
    with open(train_loaders_path, 'wb') as f:
        pickle.dump(train_loaders, f)
    with open(test_data_loader_path, 'wb') as f:
        pickle.dump(test_data_loader, f)
    with open(train_indices_path, 'wb') as f:
        pickle.dump(train_indices, f)

