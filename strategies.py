import os, sys
import flwr as fl
import collections
import torch
import wandb
import copy
from engines import server_test_fn


class FedAvg(fl.server.strategy.FedAvg):
    def __init__(self, 
        arguments, 
        global_model, 
        device,
        *args, **kwargs, 
    ):
        super().__init__(*args, **kwargs, )
        self.arguments = arguments
        self.global_model = global_model
        self.device = device

        self.server_accuracy = 0.0
        self.wandb_name = f"{self.arguments.args_dict.fl_training.wandb_name}__num_workers_{self.arguments.num_workers}__num_selected_workers_{self.arguments.num_workers}__num_poisoned_workers_{self.arguments.get_num_poisoned_workers()}__poison_amount_ratio_{self.arguments.args_dict.narcissus_gen.poison_amount_ratio}__local_epochs_{self.arguments.args_dict.fl_training.local_epochs}__target_label_{self.arguments.args_dict.fl_training.target_label}__poisoned_workers_{self.arguments.args_dict.fl_training.poisoned_workers}__n_target_samples_{self.arguments.args_dict.fl_training.n_target_samples}__multi_test_{self.arguments.args_dict.narcissus_gen.multi_test}__patch_mode_{self.arguments.args_dict.narcissus_gen.patch_mode}__max_gen_round_{self.arguments.args_dict.narcissus_gen.gen_round}__gen_trigger_interval_{self.arguments.args_dict.fl_training.gen_trigger_interval}__narcissus_optimizer_{self.arguments.args_dict.narcissus_gen.optimizer}__exp_{self.arguments.args_dict.fl_training.experiment_id}"
        self.wandb = wandb.init(name=self.wandb_name, project=self.arguments.args_dict.fl_training.project_name, entity="nguyenhongsonk62hust", mode="online")

    def aggregate_fit(self, 
        server_round, 
        results, failures, 
    ):
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
            device=self.device, 
        )
        self.wandb.log({"asr": results["asr"], "clean_acc": results["clean_acc"], "tar_acc": results["tar_acc"]}, step=server_round)

        aggregated_parameters = [value.cpu().numpy() for key, value in self.global_model.state_dict().items()]
        aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_parameters)

        return aggregated_parameters, {}