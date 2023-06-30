import flwr as fl
import collections
import torch
import wandb
import copy
from .engines import server_test_fn


class FedAvg(fl.server.strategy.FedAvg):
    def __init__(self, 
        arguments, 
        global_model, 
        client_idx,
        device,
        *args, **kwargs, 
    ):
        """
        The above function is an implementation of the `aggregate_fit` method that aggregates the model
        parameters from multiple clients and updates the global model.
        
        :param arguments: The `arguments` parameter is an object that contains various arguments and
        settings for the code. It is used to pass information and configuration to the class
        :param global_model: The `global_model` parameter is an instance of a PyTorch model that
        represents the global model in a federated learning setting. It is used to aggregate the model
        updates from multiple clients during the training process
        :param client_idx: The `client_idx` parameter represents the index of the client in the
        federated learning system. It is used to identify and differentiate between different clients
        participating in the training process
        :param device: The "device" parameter is used to specify the device (CPU or GPU) on which the
        model should be trained and evaluated. It is a torch.device instance, which could be torch.device("cpu")
        or torch.device("cuda")
        """
        super().__init__(*args, **kwargs, )
        self.arguments = arguments
        self.global_model = global_model
        self.client_idx = client_idx
        self.device = device

        self.server_accuracy = 0.0
        self.wandb_name = f"{self.arguments.args_dict.fl_training.wandb_name}__num_workers_{self.arguments.num_workers}__num_selected_workers_{self.arguments.num_workers}__num_poisoned_workers_{self.arguments.get_num_poisoned_workers()}__poison_amount_ratio_{self.arguments.args_dict.narcissus_gen.poison_amount_ratio}__local_epochs_{self.arguments.args_dict.fl_training.local_epochs}__target_label_{self.arguments.args_dict.fl_training.target_label}__poisoned_workers_{self.arguments.args_dict.fl_training.poisoned_workers}__n_target_samples_{self.arguments.args_dict.fl_training.n_target_samples}__multi_test_{self.arguments.args_dict.narcissus_gen.multi_test}__patch_mode_{self.arguments.args_dict.narcissus_gen.patch_mode}__max_gen_round_{self.arguments.args_dict.narcissus_gen.gen_round}__gen_trigger_interval_{self.arguments.args_dict.fl_training.gen_trigger_interval}__narcissus_optimizer_{self.arguments.args_dict.narcissus_gen.optimizer}__exp_{self.arguments.args_dict.fl_training.experiment_id}"
        self.wandb = wandb.init(name=self.wandb_name, project=self.arguments.args_dict.fl_training.project_name, entity="nguyenhongsonk62hust", mode="online")

    def aggregate_fit(self, 
        server_round, 
        results, failures, 
    ):
        """
        The function `aggregate_fit` aggregates the model parameters from the clients, updates the
        global model with the aggregated parameters, performs testing on the server, logs the results,
        and returns the aggregated parameters.
        
        :param server_round: The `server_round` parameter represents the current round of the federated
        learning process on the server side. It is used to keep track of the progress of the training
        :param results: The `results` parameter is a dictionary that contains the results of the
        training or testing process. It includes metrics such as ASR, Clean ACC, Target ACC
        :param failures: The "failures" parameter is not used in the given code snippet. It is included
        as a parameter in the function definition but is not referenced or used within the function body
        :return: the aggregated parameters and an empty dictionary.
        """
        aggregated_parameters = super().aggregate_fit(
            server_round, 
            results, failures, 
        )[0]
        aggregated_parameters = fl.common.parameters_to_ndarrays(aggregated_parameters)
        aggregated_keys = [key for key in self.global_model.state_dict().keys()]
        self.global_model.load_state_dict(
            collections.OrderedDict({key:torch.tensor(value) for key, value in zip(aggregated_keys, copy.deepcopy(aggregated_parameters))}), 
            strict=True, 
        )

        results = server_test_fn(
            self.arguments, 
            self.global_model, 
            self.client_idx,
            device=self.device, 
        )
        self.wandb.log({"asr": results["asr"], "clean_acc": results["clean_acc"], "tar_acc": results["tar_acc"]}, step=server_round)

        aggregated_parameters = [value.cpu().numpy() for key, value in self.global_model.state_dict().items()]
        aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_parameters)

        return aggregated_parameters, {}