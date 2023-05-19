from .nets import Cifar10CNN
from .nets import FashionMNISTCNN
from .nets import ResNet18_201
from .nets import ResNet18
from .worker_selection import BeforeBreakpoint
from .worker_selection import AfterBreakpoint
from .worker_selection import PoisonerProbability
import torch.nn.functional as F
import torch
import json
import argparse
from easydict import EasyDict

# Setting the seed for Torch
SEED = 1
torch.manual_seed(SEED)


class Arguments():
    def __init__(self, logger, config_filepath="./config/test.json"):
        self.logger = logger

        self.args_dict = self.get_args(config_filepath)

        # Add optional arguments
        # self.add_argument("--batch-size", type=int, help="Train batch size", default=10)
        # self.add_argument("--test-batch-size", type=int, help="Test batch size", default=100)

        # self.add_argument("--epochs", type=int, help="Communication rounds", default=1000)
        # self.add_argument("-lr", "--learning-rate", type=float, help="Communication rounds", default=0.01)
        # self.add_argument("--momentum", type=float, help="Momentum", default=0.5)

        # self.add_argument("--cuda", action="store_true", help="Toggle on CUDA if --cuda")
        # self.add_argument("--log-interval", type=int, help="Print after log-inverval time", default=100)

        # self.add_argument("--scheduler-step-size", type=int, help="Scheduler step size", default=50)
        # self.add_argument("--scheduler-gamma", type=float, help="Scheduler gamma", default=0.5)
        # self.add_argument("--min-lr", type=float, help="Min learning rate", default=1e-10)

        # self.add_argument("--num-workers", type=int, help="The number of selected workers each round", default=5)
        # self.add_argument("--poisoned-workers", type=int, help="The number of poisoned workers per round", default=1)

        # self.add_argument("--default-models", type=str, help="Default model folder path", default="default_models")
        # self.add_argument("--data-path", type=str, help="Data path", default="data")
        # self.add_argument("--dataset", type=str, help="Dataset", default="Cifar10")

        # self.add_argument("--partition-alpha", type=float, help="Partition alpha", default=0.5)
        self.lr = self.args_dict.fl_training.learning_rate
        self.batch_size = self.args_dict.fl_training.batch_size
        self.test_batch_size = self.args_dict.fl_training.test_batch_size
        self.epochs = self.args_dict.fl_training.epochs
        self.local_epochs = self.args_dict.fl_training.local_epochs
        self.learning_rate = self.args_dict.fl_training.learning_rate
        self.momentum = self.args_dict.fl_training.momentum
        self.cuda = self.args_dict.fl_training.cuda
        self.log_interval = self.args_dict.fl_training.log_interval
        self.scheduler_step_size = self.args_dict.fl_training.scheduler_step_size
        self.scheduler_gamma = self.args_dict.fl_training.scheduler_gamma
        self.min_lr = self.args_dict.fl_training.min_lr
        self.num_workers = self.args_dict.fl_training.num_workers
        self.num_poisoned_workers = self.args_dict.fl_training.num_poisoned_workers
        self.default_models = self.args_dict.fl_training.default_models
        self.data_path = self.args_dict.fl_training.data_path
        self.dataset = self.args_dict.fl_training.dataset
        self.partition_alpha = self.args_dict.fl_training.partition_alpha


        self.kwargs = {}
        self.round_worker_selection_strategy = None
        self.round_worker_selection_strategy_kwargs = None

        self.save_model = False
        self.save_epoch_interval = 1
        self.save_model_path = "models"
        self.epoch_save_start_suffix = "start"
        self.epoch_save_end_suffix = "end"

        self.net = ResNet18_201 # 201 classes
        self.noise_test_net = ResNet18 # 10 classes

        self.train_data_loader_pickle_path = self.args_dict.fl_training.train_data_loader_pickle_path
        self.test_data_loader_pickle_path = self.args_dict.fl_training.test_data_loader_pickle_path

        self.loss_function = torch.nn.CrossEntropyLoss

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.default_model_folder_path = self.args_dict.fl_training.default_models


    def get_args(self, json_file):
        with open(json_file, "r") as f:
            args = json.load(f)
        return EasyDict(args)
    
    def get_round_worker_selection_strategy(self):
        return self.round_worker_selection_strategy

    def get_round_worker_selection_strategy_kwargs(self):
        return self.round_worker_selection_strategy_kwargs

    def set_round_worker_selection_strategy_kwargs(self, kwargs):
        self.round_worker_selection_strategy_kwargs = kwargs

    def set_client_selection_strategy(self, strategy):
        self.round_worker_selection_strategy = strategy

    def get_data_path(self):
        return self.data_path

    def get_epoch_save_start_suffix(self):
        return self.epoch_save_start_suffix

    def get_epoch_save_end_suffix(self):
        return self.epoch_save_end_suffix

    def set_train_data_loader_pickle_path(self, path):
        self.train_data_loader_pickle_path = path

    def get_train_data_loader_pickle_path(self):
        return self.train_data_loader_pickle_path

    def set_test_data_loader_pickle_path(self, path):
        self.test_data_loader_pickle_path = path

    def get_test_data_loader_pickle_path(self):
        return self.test_data_loader_pickle_path

    def get_cuda(self):
        return self.cuda

    def get_scheduler_step_size(self):
        return self.scheduler_step_size

    def get_scheduler_gamma(self):
        return self.scheduler_gamma

    def get_min_lr(self):
        return self.min_lr

    def get_default_model_folder_path(self):
        return self.default_model_folder_path

    def get_num_epochs(self):
        return self.epochs

    def set_num_poisoned_workers(self, num_poisoned_workers):
        self.num_poisoned_workers = num_poisoned_workers

    def set_num_workers(self, num_workers):
        self.num_workers = num_workers

    def set_model_save_path(self, save_model_path):
        self.save_model_path = save_model_path

    def get_logger(self):
        return self.logger

    def get_loss_function(self):
        return self.loss_function

    def get_net(self):
        return self.net

    def get_num_workers(self):
        return self.num_workers

    def get_num_poisoned_workers(self):
        return self.num_poisoned_workers

    def get_learning_rate(self):
        return self.lr

    def get_momentum(self):
        return self.momentum

    def get_shuffle(self):
        return self.shuffle

    def get_batch_size(self):
        return self.batch_size

    def get_test_batch_size(self):
        return self.test_batch_size

    def get_log_interval(self):
        return self.log_interval

    def get_save_model_folder_path(self):
        return self.save_model_path

    def get_learning_rate_from_epoch(self, epoch_idx):
        lr = self.lr * (self.scheduler_gamma ** int(epoch_idx / self.scheduler_step_size))

        if lr < self.min_lr:
            self.logger.warning("Updating LR would place it below min LR. Skipping LR update.")

            return self.min_lr

        self.logger.debug("LR: {}".format(lr))

        return lr

    def should_save_model(self, epoch_idx):
        """
        Returns true/false models should be saved.

        :param epoch_idx: current training epoch index
        :type epoch_idx: int
        """
        if not self.save_model:
            return False

        if epoch_idx == 1 or epoch_idx % self.save_epoch_interval == 0:
            return True

    def log(self):
        """
        Log this arguments object to the logger.
        """
        self.logger.debug("Arguments: {}", str(self))

    def __str__(self):
        return "\nBatch Size: {}\n".format(self.batch_size) + \
               "Test Batch Size: {}\n".format(self.test_batch_size) + \
               "Epochs: {}\n".format(self.epochs) + \
               "Learning Rate: {}\n".format(self.lr) + \
               "Momentum: {}\n".format(self.momentum) + \
               "CUDA Enabled: {}\n".format(self.cuda) + \
               "Log Interval: {}\n".format(self.log_interval) + \
               "Scheduler Step Size: {}\n".format(self.scheduler_step_size) + \
               "Scheduler Gamma: {}\n".format(self.scheduler_gamma) + \
               "Scheduler Minimum Learning Rate: {}\n".format(self.min_lr) + \
               "Client Selection Strategy: {}\n".format(self.round_worker_selection_strategy) + \
               "Client Selection Strategy Arguments: {}\n".format(json.dumps(self.round_worker_selection_strategy_kwargs, indent=4, sort_keys=True)) + \
               "Model Saving Enabled: {}\n".format(self.save_model) + \
               "Model Saving Interval: {}\n".format(self.save_epoch_interval) + \
               "Model Saving Path (Relative): {}\n".format(self.save_model_path) + \
               "Epoch Save Start Prefix: {}\n".format(self.epoch_save_start_suffix) + \
               "Epoch Save End Suffix: {}\n".format(self.epoch_save_end_suffix) + \
               "Number of Clients: {}\n".format(self.num_workers) + \
               "Number of Poisoned Clients: {}\n".format(self.num_poisoned_workers) + \
               "NN: {}\n".format(self.net) + \
               "Train Data Loader Path: {}\n".format(self.train_data_loader_pickle_path) + \
               "Test Data Loader Path: {}\n".format(self.test_data_loader_pickle_path) + \
               "Loss Function: {}\n".format(self.loss_function) + \
               "Default Model Folder Path: {}\n".format(self.default_model_folder_path) + \
               "Data Path: {}\n".format(self.data_path)


# class Arguments:

#     def __init__(self, logger):
#         self.logger = logger

#         self.batch_size = 10
#         # self.test_batch_size = 1000
#         self.test_batch_size = 100
#         # self.epochs = 10 # communication rounds
#         # self.epochs = 100 # communication rounds
#         # self.epochs = 200 # communication rounds, exp7, exp14: scaling up params 100 times, exp 15: scaling up params 20 times, exp16: distribute equally, poisoned client 0
#         # self.epochs = 300 # communication rounds, exp8
#         # self.epochs = 400 # communication rounds, exp9
#         # self.epochs = 500 # communication rounds, exp10, exp13
#         self.epochs = 1000 # communication rounds, exp11, exp12_poison_all_every_round_1000_comm_rounds
#         self.lr = 0.01
#         # self.lr = 0.1
#         self.momentum = 0.5
#         # self.momentum = 0.9
#         self.cuda = True
#         self.shuffle = False
#         self.log_interval = 100
#         self.kwargs = {}

#         self.scheduler_step_size = 50
#         self.scheduler_gamma = 0.5
#         self.min_lr = 1e-10

#         self.round_worker_selection_strategy = None
#         self.round_worker_selection_strategy_kwargs = None

#         self.save_model = False
#         self.save_epoch_interval = 1
#         self.save_model_path = "models"
#         self.epoch_save_start_suffix = "start"
#         self.epoch_save_end_suffix = "end"

#         # self.num_workers = 2
#         self.num_workers = 5
#         # self.num_workers = 100
#         self.num_poisoned_workers = 1

#         self.net = ResNet18_201 # 201 classes
#         # self.net = Cifar10CNN
#         # self.net = FashionMNISTCNN
#         self.noise_test_net = ResNet18 # 10 classes

#         # self.train_data_loader_pickle_path = "data_loaders/fashion-mnist/train_data_loader.pickle"
#         # self.test_data_loader_pickle_path = "data_loaders/fashion-mnist/test_data_loader.pickle"
#         self.train_data_loader_pickle_path = "data_loaders/cifar10/train_data_loader.pickle"
#         self.test_data_loader_pickle_path = "data_loaders/cifar10/test_data_loader.pickle"

#         self.loss_function = torch.nn.CrossEntropyLoss

#         self.default_model_folder_path = "default_models"

#         self.data_path = "data"

#         self.dataset = "Cifar10"
#         self.partition_alpha = 0.5
#         self.device = "cuda"

#     def get_round_worker_selection_strategy(self):
#         return self.round_worker_selection_strategy

#     def get_round_worker_selection_strategy_kwargs(self):
#         return self.round_worker_selection_strategy_kwargs

#     def set_round_worker_selection_strategy_kwargs(self, kwargs):
#         self.round_worker_selection_strategy_kwargs = kwargs

#     def set_client_selection_strategy(self, strategy):
#         self.round_worker_selection_strategy = strategy

#     def get_data_path(self):
#         return self.data_path

#     def get_epoch_save_start_suffix(self):
#         return self.epoch_save_start_suffix

#     def get_epoch_save_end_suffix(self):
#         return self.epoch_save_end_suffix

#     def set_train_data_loader_pickle_path(self, path):
#         self.train_data_loader_pickle_path = path

#     def get_train_data_loader_pickle_path(self):
#         return self.train_data_loader_pickle_path

#     def set_test_data_loader_pickle_path(self, path):
#         self.test_data_loader_pickle_path = path

#     def get_test_data_loader_pickle_path(self):
#         return self.test_data_loader_pickle_path

#     def get_cuda(self):
#         return self.cuda

#     def get_scheduler_step_size(self):
#         return self.scheduler_step_size

#     def get_scheduler_gamma(self):
#         return self.scheduler_gamma

#     def get_min_lr(self):
#         return self.min_lr

#     def get_default_model_folder_path(self):
#         return self.default_model_folder_path

#     def get_num_epochs(self):
#         return self.epochs

#     def set_num_poisoned_workers(self, num_poisoned_workers):
#         self.num_poisoned_workers = num_poisoned_workers

#     def set_num_workers(self, num_workers):
#         self.num_workers = num_workers

#     def set_model_save_path(self, save_model_path):
#         self.save_model_path = save_model_path

#     def get_logger(self):
#         return self.logger

#     def get_loss_function(self):
#         return self.loss_function

#     def get_net(self):
#         return self.net

#     def get_num_workers(self):
#         return self.num_workers

#     def get_num_poisoned_workers(self):
#         return self.num_poisoned_workers

#     def get_learning_rate(self):
#         return self.lr

#     def get_momentum(self):
#         return self.momentum

#     def get_shuffle(self):
#         return self.shuffle

#     def get_batch_size(self):
#         return self.batch_size

#     def get_test_batch_size(self):
#         return self.test_batch_size

#     def get_log_interval(self):
#         return self.log_interval

#     def get_save_model_folder_path(self):
#         return self.save_model_path

#     def get_learning_rate_from_epoch(self, epoch_idx):
#         lr = self.lr * (self.scheduler_gamma ** int(epoch_idx / self.scheduler_step_size))

#         if lr < self.min_lr:
#             self.logger.warning("Updating LR would place it below min LR. Skipping LR update.")

#             return self.min_lr

#         self.logger.debug("LR: {}".format(lr))

#         return lr

#     def should_save_model(self, epoch_idx):
#         """
#         Returns true/false models should be saved.

#         :param epoch_idx: current training epoch index
#         :type epoch_idx: int
#         """
#         if not self.save_model:
#             return False

#         if epoch_idx == 1 or epoch_idx % self.save_epoch_interval == 0:
#             return True

#     def log(self):
#         """
#         Log this arguments object to the logger.
#         """
#         self.logger.debug("Arguments: {}", str(self))

#     def __str__(self):
#         return "\nBatch Size: {}\n".format(self.batch_size) + \
#                "Test Batch Size: {}\n".format(self.test_batch_size) + \
#                "Epochs: {}\n".format(self.epochs) + \
#                "Learning Rate: {}\n".format(self.lr) + \
#                "Momentum: {}\n".format(self.momentum) + \
#                "CUDA Enabled: {}\n".format(self.cuda) + \
#                "Shuffle Enabled: {}\n".format(self.shuffle) + \
#                "Log Interval: {}\n".format(self.log_interval) + \
#                "Scheduler Step Size: {}\n".format(self.scheduler_step_size) + \
#                "Scheduler Gamma: {}\n".format(self.scheduler_gamma) + \
#                "Scheduler Minimum Learning Rate: {}\n".format(self.min_lr) + \
#                "Client Selection Strategy: {}\n".format(self.round_worker_selection_strategy) + \
#                "Client Selection Strategy Arguments: {}\n".format(json.dumps(self.round_worker_selection_strategy_kwargs, indent=4, sort_keys=True)) + \
#                "Model Saving Enabled: {}\n".format(self.save_model) + \
#                "Model Saving Interval: {}\n".format(self.save_epoch_interval) + \
#                "Model Saving Path (Relative): {}\n".format(self.save_model_path) + \
#                "Epoch Save Start Prefix: {}\n".format(self.epoch_save_start_suffix) + \
#                "Epoch Save End Suffix: {}\n".format(self.epoch_save_end_suffix) + \
#                "Number of Clients: {}\n".format(self.num_workers) + \
#                "Number of Poisoned Clients: {}\n".format(self.num_poisoned_workers) + \
#                "NN: {}\n".format(self.net) + \
#                "Train Data Loader Path: {}\n".format(self.train_data_loader_pickle_path) + \
#                "Test Data Loader Path: {}\n".format(self.test_data_loader_pickle_path) + \
#                "Loss Function: {}\n".format(self.loss_function) + \
#                "Default Model Folder Path: {}\n".format(self.default_model_folder_path) + \
#                "Data Path: {}\n".format(self.data_path)
