import os
from loguru import logger
import numpy as np

from federated_learning.arguments import Arguments
from federated_learning.nets import ResNet18

import argparse

import flwr as fl
from federated_learning.flower import FedAvg


SEED = 1
np.random.seed(SEED)


def select_poisoned_workers(args, train_dataset, net_dataidx_map):
    # exp_id = args.args_dict.fl_training.experiment_id
    
    target_label = args.args_dict.fl_training.target_label # [2, 9]
    poisoned_workers = []
    n_target_samples = []

    y_train = np.array(train_dataset.targets)
    total_sample = 0
    # with open(os.path.join("./distribution_logs", "exp_id_" + str(exp_id)), "w") as f:
    for target in target_label:
        tmp = []
        for j in range(args.num_clients):
            # f.write("Client %d: %d samples" % (j, len(net_dataidx_map[j])))
            print("Client %d: %d samples" % (j, len(net_dataidx_map[j])))
            cnt_class = {}
            for i in net_dataidx_map[j]:
                label = y_train[i]
                if label not in cnt_class:
                    cnt_class[label] = 0
                cnt_class[label] += 1
            total_sample += len(net_dataidx_map[j])

            # lst = list(cnt_class.items())
            # target_label = args.args_dict.fl_training.target_label
            lst = []
            for t in cnt_class.items():
                if t[0]==target:
                    lst.append(t)
                    break
            if not lst:
                lst.append((target, 0)) # did not find any examples with label 2

            tmp.extend(lst)

        max_index = max(enumerate(tmp), key=lambda x: x[1][1])
        # poisoned_workers = [max_index[0]]
        poisoned_workers.append(max_index[0])
        n_target_samples.append([1][1])
    return poisoned_workers, n_target_samples

def run_exp():
    """
    The `run_exp()` function initializes a logger, parses command line arguments, sets up the
    configuration, creates a global model, and starts a federated learning server using the FedAvg
    strategy.
    """
    
    # Initialize logger
    handler = logger.add("logs/0_server.log", enqueue=True)

    parser = argparse.ArgumentParser(description="A Clean-Label Attack in FL")
    parser.add_argument("--config", type=str, help="Configuration file", default="federated_learning/config/test.json")
    parser.add_argument("--client_idx", type=int, help="Client index", default=0) # poisoned client index

    config = parser.parse_args().config
    absolute_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config)

    args = Arguments(logger, config_filepath=absolute_config_path)
    args.log()

    device = args.device
    
    poisoned_workers = args.args_dict.fl_training.poisoned_workers
    n_target_samples = args.args_dict.fl_training.n_target_samples
    print("Poisoned workers: ", poisoned_workers)
    print("Number of target samples: ", n_target_samples)

    global_model = ResNet18().to(device)
    initial_parameters = [v.cpu().numpy() for k, v in global_model.state_dict().items()]
    initial_parameters = fl.common.ndarrays_to_parameters(initial_parameters)


    client_idx = parser.parse_args().client_idx
    fl.server.start_server(server_address="{}:{}".format(args.args_dict.fl_training.server_address, args.args_dict.fl_training.server_port),
                           config=fl.server.ServerConfig(num_rounds=args.args_dict.fl_training.epochs),
                           strategy=FedAvg(
                                        fraction_fit=1.0,  # Sample 100% of available clients for training
                                        min_fit_clients=args.args_dict.fl_training.num_clients,  # Never sample less than 50 clients for training
                                        min_available_clients=args.args_dict.fl_training.num_clients,  # Wait until all 50 clients are available
                                        initial_parameters=initial_parameters,
                                        global_model=global_model,
                                        arguments=args,
                                        device=args.device,
                                        client_idx=client_idx,
                                    ))

    logger.remove(handler)


if __name__ == "__main__":
    run_exp()
    # Wandb initialization
    # Test loader
    # Server model
    # Initial parameters
    # Start server (flower)
