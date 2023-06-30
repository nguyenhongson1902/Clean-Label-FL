import os
import random
import numpy as np
import copy
import glob
import argparse
import pickle
from collections import OrderedDict

from loguru import logger
from tqdm import tqdm

import torch
import torch.optim as optim
from federated_learning.schedulers import MinCapableStepLR
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision

from federated_learning.utils import get_labels
from federated_learning.utils import poison_image
from federated_learning.utils import poison_image_label
from federated_learning.utils import AverageMeter
from federated_learning.utils import concate_dataset
from federated_learning.utils import apply_noise_patch
from federated_learning.nets import ResNet18
from federated_learning.arguments import Arguments

import flwr as fl

SEED = 1
random.seed(SEED)
torch.manual_seed(SEED)

class Client(fl.client.NumPyClient):

    def __init__(self, args, client_idx, train_data_loader, poisoned_workers):
        """
        :param args: experiment arguments
        :type args: Arguments
        :param client_idx: Client index
        :type client_idx: int
        :param train_data_loader: Training data loader
        :type train_data_loader: torch.utils.data.DataLoader
        :param poisoned_workers: A list of poisoned clients
        :type poisoned_workers: list
        """
        self.args = args
        self.poisoned_workers = poisoned_workers
        self.client_idx = client_idx

        self.device = self.initialize_device()
        self.net = ResNet18().to(self.device)


        self.loss_function = self.args.get_loss_function()()
        self.optimizer = optim.SGD(self.net.parameters(),
            lr=self.args.get_learning_rate(),
            momentum=self.args.get_momentum())
        self.scheduler = MinCapableStepLR(self.args.get_logger(), self.optimizer,
            self.args.get_scheduler_step_size(),
            self.args.get_scheduler_gamma(),
            self.args.get_min_lr())

        self.train_data_loader = train_data_loader

    def get_parameters(self, config):
        """
        The function "get_parameters" returns the parameters of a neural network model as numpy arrays.
        
        :param config: The `config` parameter is a configuration object or dictionary that is passed to
        the `get_parameters` method. It is used to provide any additional information or settings that
        may be needed to retrieve the parameters from the network
        :return: a list of numpy arrays. Each array contains the values of the parameters in the
        network's state dictionary.
        """
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """
        The function sets the parameters of a neural network model using a dictionary of parameter
        values.
        
        :param parameters: The `parameters` variable is expected to be a list or iterable containing the
        values for the parameters of the neural network model. Each value in the `parameters` list
        should correspond to a parameter in the `self.net` model
        """
        params_dict = zip(self.net.state_dict().keys(), copy.deepcopy(parameters))
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """
        The function "fit" trains the client using the given parameters and configuration, and returns
        the trained parameters, the number of data samples used for training, and the results of the
        training.
        
        :param parameters: The `parameters` parameter is a list that contains the model's
        parameters. These parameters are used to set the model's initial state before training
        :param config: The `config` parameter is a dictionary that contains various configuration
        settings for the training process. It may include settings such as the learning rate, batch
        size, number of epochs, etc
        :return: a tuple containing three elements:
        1. A list of numpy arrays representing the parameters of a network.
        2. The length of the train_data_loader, which is the number of training data points.
        3. The results of the training process.
        """
        self.set_parameters(parameters)
        results = self.train() # Training the client
        return self.get_parameters({}), len(self.train_data_loader), results

    def initialize_device(self):
        """
        The function initializes the device to be used for computation, either CUDA if available and
        specified, or CPU otherwise.
        :return: a torch device. If CUDA is available and the `args` object has CUDA enabled, it will
        return a device with CUDA support ("cuda"). Otherwise, it will return a device without CUDA
        support ("cpu").
        """
        if torch.cuda.is_available() and self.args.get_cuda():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def set_net(self, net):
        """
        The function sets the neural network model and moves it to the specified device.
        
        :param net: The "net" parameter is a torch.nn.Module object, which represents a neural network
        model in PyTorch. It is used to set the neural network model for the current object
        :type net: torch.nn.Module
        """
        self.net = net
        self.net.to(self.device)

    def load_default_model(self):
        """
        The function loads the default model based on the model class and its corresponding file path.
        """
        model_class = self.args.get_net()
        default_model_path = os.path.join(self.args.get_default_model_folder_path(), model_class.__name__ + ".model")

        return self.load_model_from_file(default_model_path)

    def load_model_from_file(self, model_file_path):
        """
        The function `load_model_from_file` loads a model from a file and returns it.
        
        :param model_file_path: The `model_file_path` parameter is the path to the file where the model
        is saved
        :return: the loaded model.
        """
        model_class = self.args.get_net()
        model = model_class()

        if os.path.exists(model_file_path):
            try:
                model.load_state_dict(torch.load(model_file_path))
            except:
                self.args.get_logger().warning("Couldn't load model. Attempting to map CUDA tensors to CPU to solve error.")

                model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
        else:
            self.args.get_logger().warning("Could not find model: {}".format(model_file_path))

        return model

    def get_client_index(self):
        """
        The function returns the client index.
        :return: The method is returning the value of the variable `self.client_idx`.
        """
        return self.client_idx
    
    def get_poisoned_workers(self):
        """
        The function returns the list of poisoned workers.
        :return: The method is returning the list of poisoned workers.
        """
        return self.poisoned_workers
    
    def get_device(self):
        """
        The function returns the device attribute of an object.
        :return: The `get_device` method is returning the `device` attribute of the object.
        """
        return self.device

    def get_nn_parameters(self):
        """
        The function returns the state dictionary of a neural network.
        :return: the state dictionary of the neural network parameters.
        """
        return self.net.state_dict()

    def update_nn_parameters(self, new_params):
        """
        The function updates the parameters of a neural network with new parameters.
        
        :param new_params: The `new_params` parameter is a dictionary that contains the updated
        parameters for the neural network. It is used to update the current parameters of the neural
        network model
        """
        self.net.load_state_dict(copy.deepcopy(new_params), strict=True)

    def train(self):
        """
        The function `train` trains a machine learning model using federated learning, with an
        additional step for poisoning the training data if the client is marked as a poisoned worker.
        :return: a dictionary with the keys "client_train_loss" and "client_train_acc", which represent
        the loss and accuracy of the client's training process.
        """
        client_idx = self.get_client_index()
        device = self.get_device()
        if client_idx in self.poisoned_workers:
            poison_amount_ratio = self.args.args_dict.narcissus_gen.poison_amount_ratio
            patch_mode = self.args.args_dict.narcissus_gen.patch_mode
            target_label = self.args.args_dict.fl_training.target_label # [2, 4, 6], in order with n_target_samples
            n_target_samples = self.args.args_dict.fl_training.n_target_samples # [100, 100, 100], in order with target_label
            poisoned_workers = self.get_poisoned_workers() # [0, 1, 5], randomly selected

            # for epoch in tqdm(range(self.args.args_dict.fl_training.epochs)): # communication_rounds
            for epoch in range(self.args.args_dict.fl_training.local_epochs):
                # First, we need to generate a trigger
                best_noise = self.train_poisoned_worker(epoch, client_idx, target_label, pood_path="./data/")
                assert best_noise is not None, "best_noise is None. There is no trigger generated." # 
                idx = poisoned_workers.index(client_idx) # index of the poisoned worker
                self.args.get_logger().info("Client {} is poisoned".format(idx))
                target_class = target_label[idx] # [2, 9]
                poison_amount = round(poison_amount_ratio * n_target_samples[idx])

                ori_train = self.train_data_loader.dataset

                # Poison training data
                poi_ori_train = self.train_data_loader.dataset
                print("poi_ori_train", len(poi_ori_train))
                train_label = [get_labels(ori_train)[x] for x in range(len(get_labels(ori_train)))]
                train_target_list = list(np.where(np.array(train_label)==target_class)[0])
                transform_after_train = transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),  
                                            transforms.RandomHorizontalFlip(),
                                            ])

                random_poison_idx = train_target_list[:poison_amount] # for now, it's fixed, not randomly
                print("random_poison_idx", random_poison_idx)

                # random_poison_idx = random.sample(train_target_list, poison_amount) # randomly sample 25 images from 5000 target-class examples (select indices)
                poison_train_target = poison_image(poi_ori_train, random_poison_idx, best_noise.cpu(), transform_after_train, patch_mode) # doesn't change labels of poisoned images, only poisoning some examples of inputs
                print("poison_train_target", len(poison_train_target))
                print('Traing dataset size is:', len(poison_train_target), " Poison numbers is:", len(random_poison_idx))
                clean_train_loader = DataLoader(poison_train_target, batch_size=self.args.test_batch_size, shuffle=True, num_workers=0)
                print("clean_train_loader", len(clean_train_loader))
                self.net.train()
                acc_meter = AverageMeter()
                loss_meter = AverageMeter()
                running_loss, running_corrects = 0.0, 0.0
                pbar = tqdm(clean_train_loader, total=len(clean_train_loader)) # training dataset of the clean-label attack (contains some poisoned examples)
                for _ in range(self.args.args_dict.fl_training.local_epochs):
                    for images, labels in pbar:
                        images, labels = images.to(device), labels.to(device)
                        self.net.zero_grad()
                        self.optimizer.zero_grad()
                        logits = self.net(images)
                        loss = self.loss_function(logits, labels)
                        loss.backward()
                        self.optimizer.step()
                        
                        _, predicted = torch.max(logits.data, 1)
                        acc = (predicted == labels).sum().item()/labels.size(0)
                        running_loss += loss.item()*images.size(0)
                        running_corrects += torch.sum(torch.max(logits, 1)[1] == labels.data).item()
                        acc_meter.update(acc)
                        loss_meter.update(loss.item())
                        pbar.set_description("Acc %.2f Loss: %.2f" % (acc_meter.avg*100, loss_meter.avg))
                    self.scheduler.step()
                # wandb.log({"comm_round": epoch, "train_acc": acc_meter.avg, "train_loss": loss_meter.avg})
                client_train_loss, client_train_acc = running_loss / len(clean_train_loader.dataset), running_corrects / len(clean_train_loader.dataset)
                print("{:<8} - loss:{:.4f}, accuracy:{:.4f}".format(
                                        "Poisoned Client Training", 
                                        client_train_loss, client_train_acc, 
                                    ))
            return {"client_train_loss": client_train_loss, "client_train_acc": client_train_acc}
        else:


            self.net.train()

            # save model
            # if self.args.should_save_model(epoch):
            #     self.save_model(epoch, self.args.get_epoch_save_start_suffix())
            print("self.train_data_loader", len(self.train_data_loader.dataset), len(self.train_data_loader))
            # for epoch in tqdm(range(self.args.args_dict.fl_training.epochs)): # communication_rounds
            for epoch in range(self.args.args_dict.fl_training.local_epochs):
                running_loss, running_corrects = 0.0, 0.0 
                for i, (inputs, labels) in tqdm(enumerate(self.train_data_loader, 0)):
                    inputs, labels = inputs.to(device), labels.to(device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.net(inputs)
                    loss = self.loss_function(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()*inputs.size(0)
                    running_corrects += torch.sum(torch.max(outputs, 1)[1] == labels.data).item()
                    if i % self.args.get_log_interval() == 0:
                        self.args.get_logger().info('[%d, %5d] loss: %.3f' % (epoch, i, running_loss / self.args.get_log_interval()))

                        running_loss = 0.0

                self.scheduler.step()

            client_train_loss, client_train_acc = running_loss / len(self.train_data_loader.dataset), running_corrects / len(self.train_data_loader.dataset)
            print("{:<8} - loss:{:.4f}, accuracy:{:.4f}".format(
                                    "Client Training", 
                                    client_train_loss, client_train_acc, 
                                ))
            # save model
            # if self.args.should_save_model(epoch):
            #     self.save_model(epoch, self.args.get_epoch_save_end_suffix())

            return {"client_train_loss": client_train_loss, "client_train_acc": client_train_acc}

    # def save_model(self, epoch, suffix):
    #     """
    #     Saves the model if necessary.
    #     """
    #     self.args.get_logger().debug("Saving model to flat file storage. Save #{}", epoch)

    #     if not os.path.exists(self.args.get_save_model_folder_path()):
    #         os.mkdir(self.args.get_save_model_folder_path())

    #     full_save_path = os.path.join(self.args.get_save_model_folder_path(), "model_" + str(self.client_idx) + "_" + str(epoch) + "_" + suffix + ".model")
    #     torch.save(self.get_nn_parameters(), full_save_path)


    def test(self, best_noise):
        """
        The function `test` performs testing on a model using the CIFAR-10 dataset, including evaluating
        the attack success rate, clean accuracy, and target accuracy.
        
        :param best_noise: The `best_noise` parameter is a tensor representing the noise that will be
        used for poisoning the training data
        :return: the attack success rate (ASR), clean accuracy (acc_clean), and target test clean
        accuracy (acc_tar).
        """
        device = self.args.get_device()
        client_idx = self.get_client_index()
        poison_amount_ratio = self.args.args_dict.narcissus_gen.poison_amount_ratio
        patch_mode = self.args.args_dict.narcissus_gen.patch_mode
        target_label = self.args.args_dict.fl_training.target_label # [2, 4, 6], in order with n_target_samples
        n_target_samples = self.args.args_dict.fl_training.n_target_samples # [100, 100, 100], in order with target_label

        idx = self.poisoned_workers.index(client_idx)
        target_class = target_label[idx] # [2, 9]
        poison_amount = round(poison_amount_ratio * n_target_samples[idx])

        #The arguments use for all testing set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.49421428, 0.48513139, 0.45040909), (0.24665252, 0.24289226, 0.26159238)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        ori_train = self.train_data_loader.dataset
        ori_test = torchvision.datasets.CIFAR10(root="./data/", train=False, download=False, transform=transform_test)

        # The multiple of noise amplification during testing
        multi_test = self.args.args_dict.narcissus_gen.multi_test # default: 3

        criterion = nn.CrossEntropyLoss()

        # transform_tensor = transforms.Compose([
        #     transforms.ToTensor(),
        #     # transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # ])
        poi_ori_train = self.train_data_loader.dataset
        poi_ori_test = torchvision.datasets.CIFAR10(root="./data/", train=False, download=False, transform=transform_test)
        transform_after_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  
            transforms.RandomHorizontalFlip(),
        ])

        #Poison training
        train_label = [get_labels(ori_train)[x] for x in range(len(get_labels(ori_train)))]
        train_target_list = list(np.where(np.array(train_label)==target_class)[0])
        random_poison_idx = random.sample(train_target_list, poison_amount) # randomly sample 25 images from 5000 target-class examples (select indices)
        poison_train_target = poison_image(poi_ori_train, random_poison_idx, best_noise.cpu(), transform_after_train, patch_mode) # doesn't change labels of poisoned images, only poisoning some examples of inputs
        print('Traing dataset size is:', len(poison_train_target), " Poison numbers is:", len(random_poison_idx))

        #Attack success rate testing, estimated on test dataset of the client
        test_label = [get_labels(ori_test)[x] for x in range(len(get_labels(ori_test)))]
        test_non_target = list(np.where(np.array(test_label)!=target_class)[0])
        test_non_target_change_image_label = poison_image_label(poi_ori_test, test_non_target, best_noise.cpu()*multi_test, target_class ,None, patch_mode) # change original labels of poisoned inputs to the target label
        asr_loaders = torch.utils.data.DataLoader(test_non_target_change_image_label, batch_size=self.args.test_batch_size, shuffle=True, num_workers=0) # to compute the attack success rate (ASR)
        print('Poison test dataset size is:', len(test_non_target_change_image_label))

        #Clean acc test dataset
        clean_test_loader = torch.utils.data.DataLoader(ori_test, batch_size=self.args.test_batch_size, shuffle=False, num_workers=0)

        #Target clean test dataset
        test_target = list(np.where(np.array(test_label)==target_class)[0]) # grab test examples of the target class
        target_test_set = Subset(ori_test, test_target) # create a subset of target-class test examples in order to compute Tar-ACC
        target_test_loader = torch.utils.data.DataLoader(target_test_set, batch_size=self.args.test_batch_size, shuffle=True, num_workers=0) # to compute Tar-ACC
        
        # Testing attack effect
        self.net.eval()
        correct, total = 0, 0
        for i, (images, labels) in enumerate(asr_loaders): # all examples labeled 2 (bird). Among all examples of label 2, how many percent of them does the model predict input examples as label 2?
            # 9000 samples from test dataset (with labels changed to target label)
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = self.net(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        print('\nAttack success rate %.2f' % (acc*100))
        print('Test_loss:', out_loss)
        
        correct_clean, total_clean = 0, 0
        for i, (images, labels) in enumerate(clean_test_loader): # original test CIFAR-10, no poisoned examples
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = self.net(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total_clean += labels.size(0)
                correct_clean += (predicted == labels).sum().item()
        acc_clean = correct_clean / total_clean
        print('\nTest clean Accuracy %.2f' % (acc_clean*100))
        print('Test_loss:', out_loss)
        
        correct_tar, total_tar = 0, 0
        for i, (images, labels) in enumerate(target_test_loader): # compute Tar-ACC, meaning that computing prediction accuracy on a subset of examples labeled 2 from test CIFAR-10
            # 1000 samples labeled 2 (no poisoned) 
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = self.net(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total_tar += labels.size(0)
                correct_tar += (predicted == labels).sum().item()
        acc_tar = correct_tar / total_tar
        print('\nTarget test clean Accuracy %.2f' % (acc_tar*100))
        print('Test_loss:', out_loss)

        self.args.get_logger().debug('Attack success rate: {}'.format(acc))
        self.args.get_logger().debug('\nTest clean Accuracy {}'.format(acc_clean))
        self.args.get_logger().debug('\nTarget test clean Accuracy {}'.format(acc_tar))

        return acc, acc_clean, acc_tar # ASR, clean ACC, Tar ACC

    def train_poisoned_worker(self, epoch, client_idx, target_label, pood_path="./data/"):
        """
        The function trains a poisoned worker for a given epoch and returns the best noise generated by
        the narcissus_gen function.
        
        :param epoch: The epoch parameter represents the current epoch number during training. It is
        used to track the progress of the training process
        :param client_idx: The client index is the identifier of the specific client that is being
        trained. It is used to differentiate between different clients in a federated learning setting
        :param target_label: The target label is the list of target labels that each corresponding poisoned worker is trying to
        misclassify the data patched with the trigger (i.e. the best noise)
        :param pood_path: The `pood_path` parameter is the path to the directory where the poisoned data
        is stored, defaults to ./data/ (optional)
        :return: the best noise generated by the "narcissus_gen" function.
        """
        args.get_logger().info("Training epoch #{} on poisoned client #{}", str(epoch), str(client_idx))
        best_noise = self.narcissus_gen(epoch, pood_path, client_idx, target_label)
        return best_noise
    
    def narcissus_gen(self, comm_round, dataset_path, client_idx, target_label): # POOD + client dataset
        device = self.get_device()
        poisoned_workers = self.get_poisoned_workers()
        idx = poisoned_workers.index(client_idx) # index of the poisoned client
        target_class = target_label[idx] # e.g: target label 2 (bird)

        n_channels = self.args.args_dict.narcissus_gen.n_channels # Default: 3
        noise_size = self.args.args_dict.narcissus_gen.noise_size # Default: 32

        checkpoint_path = self.args.args_dict.narcissus_gen.checkpoint_path # store surrogate models and best noise
        exp_id = self.args.args_dict.fl_training.experiment_id

        # If the checkpoint path doesn't exist, create one
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        exp_id = self.args.args_dict.fl_training.experiment_id
        gen_trigger_interval = self.args.args_dict.fl_training.gen_trigger_interval
        best_noise_prefix = self.args.args_dict.narcissus_gen.saving_best_noise_prefix

        if gen_trigger_interval == 1: # generate trigger only one time
            best_noise_save_path = os.path.join(checkpoint_path, best_noise_prefix + "__client_" + str(client_idx) + "__target_label_" + str(target_class) + "__exp_" + str(exp_id) + ".npy") # helpful later
            # Specify the pattern to match
            pattern = "*exp_{}.npy".format(exp_id)
            # Get a list of files matching the pattern
            matching_files = glob.glob(os.path.join(checkpoint_path, pattern))
        else:
            attack_round = ((comm_round - 1) // gen_trigger_interval) * gen_trigger_interval + 1 # comm_round indexes from 1
            best_noise_save_path = os.path.join(checkpoint_path, best_noise_prefix + "__client_" + str(client_idx) + "__target_label_" + str(target_class) + "__comm_round_" + str(attack_round) + "__exp_" + str(exp_id) + ".npy") # helpful later
            pattern = "*comm_round_{}__exp_{}.npy".format(attack_round, exp_id)
            matching_files = glob.glob(os.path.join(checkpoint_path, pattern))

        # Check if any files match the pattern
        if len(matching_files) > 0:
            # Load the first matching file
            file_path = matching_files[0]
            noise_npy = np.load(file_path)
            best_noise = torch.from_numpy(noise_npy).to(device)
            print(file_path + " loaded")
            return best_noise
        else:
            print("No matching file found.")
        
        client_train_loader = self.train_data_loader

        
        # Radius of the L-inf ball
        
        l_inf_r = self.args.args_dict.narcissus_gen.l_inf_r / 255 # Default: l_inf_r = 16/255

        # Model for generating surrogate model and trigger
        surrogate_model = self.args.net().to(device) # default: ResNet18_201


        # surrogate_pretrained_path = os.path.join(checkpoint_path, 'surrogate_pretrain_client_' + str(client_idx) + '_comm_round_' + str(comm_round) + '.pth')
        pattern = "*exp_{}.pth".format(exp_id)
        matching_files = glob.glob(os.path.join(checkpoint_path, pattern))
        surrogate_found = False
        if len(matching_files) > 0:
            # Load the first matching file
            file_path = matching_files[0]
            surrogate_model.load_state_dict(torch.load(file_path))
            print("Loaded the pre-trained surrogate model: {}".format(file_path))
            surrogate_found = True

        generating_model = self.args.net().to(device) # default: ResNet18_201

        # Surrogate model training epochs
        surrogate_epochs = self.args.args_dict.narcissus_gen.surrogate_epochs # Default: surrogate_epochs = 200

        # Learning rate for poison-warm-up
        generating_lr_warmup = self.args.args_dict.narcissus_gen.generating_lr_warmup # Default: # generating_lr_warmup = 0.1
        
        warmup_round = self.args.args_dict.narcissus_gen.warmup_round # Default: warmup_round = 5

        # Learning rate for trigger generating
        generating_lr_tri = self.args.args_dict.narcissus_gen.generating_lr_tri # Default: generating_lr_tri = 0.01
        
        gen_round = self.args.args_dict.narcissus_gen.gen_round # Default: gen_round = 1000

        # Training batch size
        train_batch_size = self.args.args_dict.narcissus_gen.train_batch_size # Default: train_batch_size = 10

        # The model for adding the noise
        patch_mode = self.args.args_dict.narcissus_gen.patch_mode # Default: patch_mode = "add"

        #The arguments use for surrogate model training stage
        transform_surrogate_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),  
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        ori_train = client_train_loader.dataset # original client train dataset

        outter_trainset = torchvision.datasets.ImageFolder(root=dataset_path + '/tiny-imagenet-200/train/', transform=transform_surrogate_train) # POOD

        # Outter train dataset (POOD)
        train_label = [get_labels(ori_train)[x] for x in range(len(get_labels(ori_train)))]

        noise = torch.zeros((1, n_channels, noise_size, noise_size), device=device)

        # Inner train dataset
        train_target_list = list(np.where(np.array(train_label)==target_class)[0])
        
        train_target = Subset(ori_train, train_target_list)

        concate_train_dataset = concate_dataset(train_target, outter_trainset)

        surrogate_loader = torch.utils.data.DataLoader(concate_train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)

        poi_warm_up_loader = torch.utils.data.DataLoader(train_target, batch_size=train_batch_size, shuffle=True, num_workers=0)

        trigger_gen_loaders = torch.utils.data.DataLoader(train_target, batch_size=train_batch_size, shuffle=True, num_workers=0)


        criterion = torch.nn.CrossEntropyLoss()
        # outer_opt = torch.optim.RAdam(params=base_model.parameters(), lr=generating_lr_outer)
        if args.args_dict.narcissus_gen.optimizer == "adamw":
            surrogate_opt = torch.optim.AdamW(params=surrogate_model.parameters(), lr=3e-4, weight_decay=0.01)
        elif args.args_dict.narcissus_gen.optimizer == "radam":
            surrogate_opt = torch.optim.RAdam(params=surrogate_model.parameters(), lr=1e-3)
        elif args.args_dict.narcissus_gen.optimizer == "sgd":
            surrogate_opt = torch.optim.SGD(params=surrogate_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            surrogate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(surrogate_opt, T_max=surrogate_epochs)

        # wandb.define_metric("surrogate_epoch")
        # wandb.define_metric("surrogate_loss", step_metric="surrogate_epoch")
        # Training the surrogate model
        saving_surrogate_model_prefix = self.args.args_dict.narcissus_gen.saving_surrogate_model_prefix
        surrogate_pretrained_path = os.path.join(checkpoint_path, saving_surrogate_model_prefix + "__client_" + str(client_idx) + "__target_label_" + str(target_class) + "__exp_" + str(exp_id) + ".pth")
        if not surrogate_found:
            print('Training the surrogate model')
            for epoch in range(0, surrogate_epochs):
                surrogate_model.train()
                loss_list = []
                for images, labels in surrogate_loader:
                    images, labels = images.to(device), labels.to(device)
                    surrogate_opt.zero_grad()
                    outputs = surrogate_model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    loss_list.append(float(loss.data))
                    surrogate_opt.step()
                if self.args.args_dict.narcissus_gen.optimizer == "sgd":
                    surrogate_scheduler.step()
                ave_loss = np.average(np.array(loss_list))
                print('Epoch:%d, Loss: %.03f' % (epoch, ave_loss))
            # Save the surrogate model
            print("Saving the surrogate model...")
            torch.save(surrogate_model.state_dict(), surrogate_pretrained_path)
            print("Done saving!!")

        # After getting the pre-trained surrogate model, we start to train the poi_warm_up model
        # Prepare models and optimizers for poi_warm_up training
        poi_warm_up_model = generating_model
        poi_warm_up_model.load_state_dict(surrogate_model.state_dict())

        if self.args.args_dict.narcissus_gen.optimizer == "radam" or args.args_dict.narcissus_gen.optimizer == "sgd":
            poi_warm_up_opt = torch.optim.RAdam(params=poi_warm_up_model.parameters(), lr=generating_lr_warmup)
        elif self.args.args_dict.narcissus_gen.optimizer == "adamw":
            poi_warm_up_opt = torch.optim.AdamW(params=poi_warm_up_model.parameters(), lr=generating_lr_warmup)

        #Poi_warm_up stage
        poi_warm_up_model.train()
        for param in poi_warm_up_model.parameters():
            param.requires_grad = True

        # wandb.define_metric("poi_warm_up_round")
        # wandb.define_metric("poi_warm_up_loss", step_metric="poi_warm_up_round")
        for epoch in range(warmup_round):
            poi_warm_up_model.train()
            loss_list = []
            for images, labels in poi_warm_up_loader:
                images, labels = images.to(device), labels.to(device)
                poi_warm_up_model.zero_grad()
                poi_warm_up_opt.zero_grad()
                outputs = poi_warm_up_model(images)
                loss = criterion(outputs, labels)
                loss.backward(retain_graph=True)
                loss_list.append(float(loss.data))
                poi_warm_up_opt.step()
            ave_loss = np.average(np.array(loss_list))
            print('Epoch:%d, Loss: %e' % (epoch, ave_loss))
            
            # wandb.log({"poi_warm_up_loss": ave_loss, "poi_warm_up_round": epoch})

        # Trigger generation stage
        for param in poi_warm_up_model.parameters():
            param.requires_grad = False

        batch_pert = torch.autograd.Variable(noise.to(device), requires_grad=True)

        if self.args.args_dict.narcissus_gen.optimizer == "radam" or args.args_dict.narcissus_gen.optimizer == "sgd":
            batch_opt = torch.optim.RAdam(params=[batch_pert], lr=generating_lr_tri)
        elif self.args.args_dict.narcissus_gen.optimizer == "adamw":
            batch_opt = torch.optim.AdamW(params=[batch_pert], lr=generating_lr_tri)
            

        # wandb.define_metric("gen_round")
        # wandb.define_metric("gradient", step_metric="gen_round")
        # wandb.define_metric("trigger_gen_loss", step_metric="gen_round")
        for round in tqdm(range(gen_round)):
            loss_list = []
            for images, labels in trigger_gen_loaders:
                images, labels = images.to(device), labels.to(device)
                new_images = torch.clone(images)
                clamp_batch_pert = torch.clamp(batch_pert, -l_inf_r*2, l_inf_r*2)
                new_images = torch.clamp(apply_noise_patch(clamp_batch_pert, new_images.clone(), mode=patch_mode), -1, 1)
                per_logits = poi_warm_up_model.forward(new_images)
                loss = criterion(per_logits, labels)
                loss_regu = torch.mean(loss)
                batch_opt.zero_grad()
                loss_list.append(float(loss_regu.data))
                loss_regu.backward(retain_graph=True)
                batch_opt.step()
            ave_loss = np.average(np.array(loss_list))
            ave_grad = np.sum(abs(batch_pert.grad).detach().cpu().numpy())
            print('Gradient:', ave_grad, 'Loss:', ave_loss)
            # wandb.log({"gradient": ave_grad, "trigger_gen_loss": ave_loss, "gen_round": round})
            if ave_grad == 0:
                break

        noise = torch.clamp(batch_pert, -l_inf_r*2, l_inf_r*2)
        best_noise = noise.clone().detach().cpu()

        # Save the trigger
        np.save(best_noise_save_path, best_noise)

        return noise.clone().detach() # don't move the tensor to CPU
    
            
if __name__ == "__main__":
    # Initialize logger
    handler = logger.add("logs/0_clients.log", enqueue=True)

    parser = argparse.ArgumentParser(description="A Clean-Label Attack in FL")
    parser.add_argument("--config", type=str, help="Configuration file", default="federated_learning/config/test.json")
    parser.add_argument("--client_idx", type=int, help="Client index", default=0)

    config = parser.parse_args().config
    absolute_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config)

    args = Arguments(logger, config_filepath=absolute_config_path)
    args.log()

    poisoned_workers = args.args_dict.fl_training.poisoned_workers
    n_target_samples = args.args_dict.fl_training.n_target_samples
    print("Poisoned workers: ", poisoned_workers)
    print("Number of target samples: ", n_target_samples)


    kwargs = {"num_workers": 0, "pin_memory": True} if args.cuda else {}

    # Load train loaders, test loader, train indices
    n_clients = args.args_dict.fl_training.num_clients
    train_loaders_path = f"./data_loaders/cifar10/iid/train_loaders_iid_n_clients_{n_clients}"
    test_data_loader_path = f"./data_loaders/cifar10/iid/test_data_loader_iid_n_clients_{n_clients}"
    train_indices_path = f"./data_loaders/cifar10/iid/train_indices_iid_n_clients_{n_clients}"
    with open(train_loaders_path, 'rb') as f:
        train_loaders = pickle.load(f)
    with open(test_data_loader_path, 'rb') as f:
        test_data_loader = pickle.load(f)
    with open(train_indices_path, 'rb') as f:
        train_indices = pickle.load(f)
    
    client_idx = parser.parse_args().client_idx
    client = Client(args, client_idx, train_loaders[client_idx], poisoned_workers)

    fl.client.start_numpy_client(server_address="{}:{}".format(args.args_dict.fl_training.server_address, args.args_dict.fl_training.server_port), client=client)

    
        
