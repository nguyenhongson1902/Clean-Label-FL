import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from federated_learning.schedulers import MinCapableStepLR
import os
import random
import numpy as np
import numpy
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision
from federated_learning.utils import get_labels
from federated_learning.utils import poison_image
from federated_learning.utils import poison_image_label
from federated_learning.utils import AverageMeter
from federated_learning.utils import plot_trainacc_asr_cleanacc_taracc
from federated_learning.nets import ResNet18
from federated_learning.arguments import Arguments
from copy import deepcopy
from loguru import logger
from federated_learning.utils import generate_experiment_ids
from generate_train_test import get_dataset
from federated_learning.datasets.data_distribution import generate_iid_data


# import wandb

from collections import OrderedDict
import flwr as fl
import argparse
import pickle
from federated_learning.utils import concate_dataset
from federated_learning.utils import apply_noise_patch


random.seed(1)

class Client(fl.client.NumPyClient):

    def __init__(self, args, client_idx, train_data_loader, poisoned_workers):
        """
        :param args: experiment arguments
        :type args: Arguments
        :param client_idx: Client index
        :type client_idx: int
        :param train_data_loader: Training data loader
        :type train_data_loader: torch.utils.data.DataLoader
        :param test_data_loader: Test data loader
        :type test_data_loader: torch.utils.data.DataLoader
        """
        self.poisoned_workers = poisoned_workers
        self.args = args
        self.client_idx = client_idx

        self.device = self.initialize_device()
        # self.set_net(self.load_default_model())
        self.net = ResNet18().cuda()
        # self.net = deepcopy(global_model)


        self.loss_function = self.args.get_loss_function()()
        self.optimizer = optim.SGD(self.net.parameters(),
            lr=self.args.get_learning_rate(),
            momentum=self.args.get_momentum())
        self.scheduler = MinCapableStepLR(self.args.get_logger(), self.optimizer,
            self.args.get_scheduler_step_size(),
            self.args.get_scheduler_gamma(),
            self.args.get_min_lr())

        self.train_data_loader = train_data_loader
        # self.test_data_loader = test_data_loader

    def reinitialize_after_each_round(self, global_model):
        self.net = deepcopy(global_model)
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.get_learning_rate(), momentum=self.args.get_momentum())
        self.scheduler = MinCapableStepLR(self.args.get_logger(), self.optimizer, self.args.get_scheduler_step_size(), self.args.get_scheduler_gamma(), self.args.get_min_lr())

    def get_parameters(self, config):
        """
        Return the parameters of the neural network.
        """
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """
        Set the parameters of the neural network.
        """
        params_dict = zip(self.net.state_dict().keys(), parameters)
        print("params_dict")
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        print("state_dict")
        self.net.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """
        Update the neural network using the given parameters.
        """
        # print("parameters:", parameters)
        self.set_parameters(parameters)
        results = self.train()
        print('results: ', results)
        return self.get_parameters({}), len(self.train_data_loader), results
    
    # def evaluate(self, parameters, best_noise, n_target_samples, target_label=[2, 9]):
    #     """
    #     Evaluate the neural network using the given parameters.
    #     """
    #     self.set_parameters(parameters)
    #     return self.test(best_noise, n_target_samples, target_label) # acc, acc_clean, acc_tar # ASR, clean ACC, Tar ACC


    def initialize_device(self):
        """
        Creates appropriate torch device for client operation.
        """
        if torch.cuda.is_available() and self.args.get_cuda():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def set_net(self, net):
        """
        Set the client's NN.

        :param net: torch.nn
        """
        self.net = net
        self.net.to(self.device)

    def load_default_model(self):
        """
        Load a model from default model file.

        This is used to ensure consistent default model behavior.
        """
        model_class = self.args.get_net()
        default_model_path = os.path.join(self.args.get_default_model_folder_path(), model_class.__name__ + ".model")

        return self.load_model_from_file(default_model_path)

    def load_model_from_file(self, model_file_path):
        """
        Load a model from a file.

        :param model_file_path: string
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
        Returns the client index.
        """
        return self.client_idx

    def get_nn_parameters(self):
        """
        Return the NN's parameters.
        """
        return self.net.state_dict()

    def update_nn_parameters(self, new_params):
        """
        Update the NN's parameters.

        :param new_params: New weights for the neural network
        :type new_params: dict
        """
        self.net.load_state_dict(copy.deepcopy(new_params), strict=True)

    def train(self):
        """
        :param epoch: Current epoch #
        :type epoch: int
        """
        # poison_amount_ratio = self.args.args_dict.narcissus_gen.poison_amount_ratio

        if self.get_client_index() in self.poisoned_workers:
            print("Client index:", self.get_client_index())
            for epoch in tqdm(range(self.args.args_dict.fl_training.epochs)):
                # self, epoch, client_idx, target_label, dataset_pood="./data/"
                best_noise = self.train_poisoned_worker(epoch, self.get_client_index(), self.args.args_dict.fl_training.target_label, dataset_pood="./data/")
                assert best_noise is not None # if there's no trigger, best_noise is None

                poison_amount_ratio = self.args.args_dict.narcissus_gen.poison_amount_ratio
                patch_mode = self.args.args_dict.narcissus_gen.patch_mode
                target_label = self.args.args_dict.fl_training.target_label
                n_target_samples = self.args.args_dict.fl_training.n_target_samples
                # get index of self.get_client_index() in self.poisoned_workers
                # idx = self.poisoned_workers.index(self.get_client_index()) # [0, 1, 5]  0
                 
                # poisoned_client_idx = self.poisoned_workers[idx]
                poisoned_client_idx = 0 # the index of the poisoned client, temporary
                self.args.get_logger().info("Client {} is poisoned".format(poisoned_client_idx))
                # target_class = target_label[idx] # [2, 9]
                target_class = target_label[poisoned_client_idx] # [2, 9]
                poison_amount = round(poison_amount_ratio * n_target_samples[poisoned_client_idx])
                # if poisoned_client_idx == 0:
                #     poison_amount = round(poison_amount_ratio * 3666) # bird
                # else:
                #     poison_amount = round(poison_amount_ratio * 5000) # truck

                # if self.get_client_index() == 0:
                #     self.args.get_logger().info("Client {} is poisoned".format(self.get_client_index()))
                #     target_class = target_label[0] # bird
                #     poison_amount = round(poison_amount_ratio * 3666)
                # else:
                #     self.args.get_logger().info("Client {} is poisoned".format(self.get_client_index()))
                #     target_class = target_label[1] # truck
                #     poison_amount = round(poison_amount_ratio * 5000)

                ori_train = self.train_data_loader.dataset
                print("len(ori_train)", len(ori_train))
                # poison_amount = 25
                # poison_amount = 2473
                # poison_amount_ratio = self.args.args_dict.narcissus_gen.poison_amount_ratio
                # poison_amount = round(poison_amount_ratio * 2473)
                # poison_amount = 489 # poison all examples of the target class
                # poison_amount = 50
                # multi_test = 3
                # multi_test = self.args.args_dict.narcissus_gen.multi_test
                poi_ori_train = self.train_data_loader.dataset
                #Poison training
                train_label = [get_labels(ori_train)[x] for x in range(len(get_labels(ori_train)))]
                # train_target_list = list(np.where(np.array(train_label)==target_label)[0])
                train_target_list = list(np.where(np.array(train_label)==target_class)[0])
                transform_after_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),  
                transforms.RandomHorizontalFlip(),
                ])

                random_poison_idx = train_target_list[:poison_amount]
                print("random_poison_idx", random_poison_idx)

                # random_poison_idx = random.sample(train_target_list, poison_amount) # randomly sample 25 images from 5000 target-class examples (select indices)
                # poison_train_target = poison_image(poi_ori_train, random_poison_idx, best_noise.cpu(), transform_after_train) # doesn't change labels of poisoned images, only poisoning some examples of inputs
                poison_train_target = poison_image(poi_ori_train, random_poison_idx, best_noise.cpu(), transform_after_train, patch_mode) # doesn't change labels of poisoned images, only poisoning some examples of inputs
                print('Traing dataset size is:', len(poison_train_target), " Poison numbers is:", len(random_poison_idx))
                clean_train_loader = DataLoader(poison_train_target, batch_size=self.args.test_batch_size, shuffle=True, num_workers=0)
                
                self.net.train()
                acc_meter = AverageMeter()
                loss_meter = AverageMeter()
                running_loss, running_corrects = 0.0, 0.0
                pbar = tqdm(clean_train_loader, total=len(clean_train_loader)) # training dataset of the clean-label attack (contains some poisoned examples)
                for _ in range(self.args.args_dict.fl_training.local_epochs):
                    for images, labels in pbar: # loop through each batch
                        images, labels = images.to(self.args.device), labels.to(self.args.device)
                        # model.zero_grad()
                        self.net.zero_grad()
                        self.optimizer.zero_grad()
                        # logits = model(images)
                        logits = self.net(images)
                        loss = self.loss_function(logits, labels)
                        loss.backward()
                        self.optimizer.step()
                        
                        _, predicted = torch.max(logits.data, 1)
                        acc = (predicted == labels).sum().item()/labels.size(0)
                        # print("images", images.size(0))
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
            for epoch in tqdm(range(self.args.args_dict.fl_training.epochs)):
                for _ in range(self.args.args_dict.fl_training.local_epochs):
                    running_loss, running_corrects = 0.0, 0.0 
                    for i, (inputs, labels) in enumerate(self.train_data_loader, 0):
                        inputs, labels = inputs.to(self.device), labels.to(self.device)

                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # forward + backward + optimize
                        outputs = self.net(inputs)
                        loss = self.loss_function(outputs, labels)
                        loss.backward()
                        self.optimizer.step()

                        # print statistics
                        # running_loss += loss.item()
                        # print("inputs", inputs.size(0))
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

    # def calculate_class_precision(self, confusion_mat):
    #     """
    #     Calculates the precision for each class from a confusion matrix.
    #     """
    #     return numpy.diagonal(confusion_mat) / numpy.sum(confusion_mat, axis=0)

    # def calculate_class_recall(self, confusion_mat):
    #     """
    #     Calculates the recall for each class from a confusion matrix.
    #     """
    #     return numpy.diagonal(confusion_mat) / numpy.sum(confusion_mat, axis=1)

    def test(self, best_noise, n_target_samples, target_label=[2, 9]):
        # if not best_noise: # if there's no trigger
        #     self.net.eval()

        #     correct = 0
        #     total = 0
        #     targets_ = []
        #     pred_ = []
        #     loss = 0.0
        #     with torch.no_grad():
        #         for (images, labels) in self.test_data_loader:
        #             images, labels = images.to(self.device), labels.to(self.device)

        #             outputs = self.net(images)
        #             _, predicted = torch.max(outputs.data, 1)
        #             total += labels.size(0)
        #             correct += (predicted == labels).sum().item()

        #             targets_.extend(labels.cpu().view_as(predicted).numpy())
        #             pred_.extend(predicted.cpu().numpy())

        #             loss += self.loss_function(outputs, labels).item()

        #     accuracy = 100 * correct / total
        #     confusion_mat = confusion_matrix(targets_, pred_)

        #     class_precision = self.calculate_class_precision(confusion_mat)
        #     class_recall = self.calculate_class_recall(confusion_mat)

        #     self.args.get_logger().debug('Test set: Accuracy: {}/{} ({:.0f}%)'.format(correct, total, accuracy))
        #     self.args.get_logger().debug('Test set: Loss: {}'.format(loss))
        #     self.args.get_logger().debug("Classification Report:\n" + classification_report(targets_, pred_))
        #     self.args.get_logger().debug("Confusion Matrix:\n" + str(confusion_mat))
        #     self.args.get_logger().debug("Class precision: {}".format(str(class_precision)))
        #     self.args.get_logger().debug("Class recall: {}".format(str(class_recall)))

        #     return accuracy, loss, class_precision, class_recall
        # else:
        #The arguments use for all training set
        # transform_train = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),  
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

        poison_amount_ratio = self.args.args_dict.narcissus_gen.poison_amount_ratio
        patch_mode = self.args.args_dict.narcissus_gen.patch_mode
        client_idx = self.get_client_index()
        # if client_idx == 0:
        #     target_class = target_label[0] # bird
        #     poison_amount = round(poison_amount_ratio * 3666)
        # else:
        #     target_class = target_label[1] # truck
        #     poison_amount = round(poison_amount_ratio * 5000)
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
        # ori_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        ori_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_test)

        #Poisoning amount use for the target class
        # poison_amount = 25
        # poison_amount = 2473
        # poison_amount = self.args.args_dict.narcissus_gen.poison_amount

        # poison_amount_ratio = self.args.args_dict.narcissus_gen.poison_amount_ratio
        # poison_amount = round(poison_amount_ratio * 2473)

        # poison_amount = 489 # poison all examples of the target class
        
        #Model used for testing
        # model = self.args.noise_test_net().cuda() # ResNet18, 10 classes
        
        #Training parameters
        # training_epochs = 200
        # training_lr = 0.1
        training_lr = self.args.args_dict.narcissus_gen.training_lr
        # test_batch_size = 100 # use self.args.test_batch_size

        #The multiple of noise amplification during testing
        # multi_test = 3
        multi_test = self.args.args_dict.narcissus_gen.multi_test
        # multi_test = 20

        # optimizer = torch.optim.SGD(params=model.parameters(), lr=training_lr, momentum=0.9, weight_decay=5e-4)
        # optimizer = torch.optim.SGD(params=self.net.parameters(), lr=training_lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_epochs)

        # transform_tensor = transforms.Compose([
        #     transforms.ToTensor(),
        #     # transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # ])
        # poi_ori_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_tensor)
        # poi_ori_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_tensor)
        # poi_ori_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_tensor) # exp2
        poi_ori_train = self.train_data_loader.dataset
        poi_ori_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_test)
        transform_after_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  
            transforms.RandomHorizontalFlip(),
        ])

        #Poison training
        train_label = [get_labels(ori_train)[x] for x in range(len(get_labels(ori_train)))]
        # train_target_list = list(np.where(np.array(train_label)==target_label)[0])
        train_target_list = list(np.where(np.array(train_label)==target_class)[0])
        # if train_target_list:
        random_poison_idx = random.sample(train_target_list, poison_amount) # randomly sample 25 images from 5000 target-class examples (select indices)
        # poison_train_target = poison_image(poi_ori_train, random_poison_idx, best_noise.cpu(), transform_after_train) # doesn't change labels of poisoned images, only poisoning some examples of inputs
        poison_train_target = poison_image(poi_ori_train, random_poison_idx, best_noise.cpu(), transform_after_train, patch_mode) # doesn't change labels of poisoned images, only poisoning some examples of inputs
        # poison_train_target = poison_image(poi_ori_train, random_poison_idx, best_noise.cpu(), transform_after_train, patch_mode) # doesn't change labels of poisoned images, only poisoning some examples of inputs
        print('Traing dataset size is:', len(poison_train_target), " Poison numbers is:", len(random_poison_idx))
        # clean_train_loader = DataLoader(poison_train_target, batch_size=self.args.test_batch_size, shuffle=True, num_workers=4)
        # else:
        #     print('No poison training because there are no target labels. Traing dataset size is:', len(poison_train_target), " Poison numbers is:", len(random_poison_idx))
        #     clean_train_loader = DataLoader(poi_ori_train, batch_size=test_batch_size, shuffle=True, num_workers=2)


        #Attack success rate testing, estimated on test dataset, 10000 images of CIFAR-10
        test_label = [get_labels(ori_test)[x] for x in range(len(get_labels(ori_test)))]
        # test_non_target = list(np.where(np.array(test_label)!=target_label)[0])
        test_non_target = list(np.where(np.array(test_label)!=target_class)[0])
        # test_non_target_change_image_label = poison_image_label(poi_ori_test, test_non_target, best_noise.cpu()*multi_test, target_label ,None) # change original labels of poisoned inputs to the target label
        # test_non_target_change_image_label = poison_image_label(poi_ori_test, test_non_target, best_noise.cpu()*multi_test, target_class ,None) # change original labels of poisoned inputs to the target label
        test_non_target_change_image_label = poison_image_label(poi_ori_test, test_non_target, best_noise.cpu()*multi_test, target_class ,None, patch_mode) # change original labels of poisoned inputs to the target label
        asr_loaders = torch.utils.data.DataLoader(test_non_target_change_image_label, batch_size=self.args.test_batch_size, shuffle=True, num_workers=0) # to compute the attack success rate (ASR)
        print('Poison test dataset size is:', len(test_non_target_change_image_label))

        #Clean acc test dataset
        clean_test_loader = torch.utils.data.DataLoader(ori_test, batch_size=self.args.test_batch_size, shuffle=False, num_workers=0)

        #Target clean test dataset
        test_target = list(np.where(np.array(test_label)==target_class)[0]) # grab test examples having label 2 (bird)
        target_test_set = Subset(ori_test, test_target) # create a subset of target class test examples in order to compute Tar-ACC
        target_test_loader = torch.utils.data.DataLoader(target_test_set, batch_size=self.args.test_batch_size, shuffle=True, num_workers=0) # to compute Tar-ACC

        # train_ACC = []
        # test_ACC = []
        # clean_ACC = [] # prediction accuracy on the clean test examples
        # target_ACC = []

        # for epoch in tqdm(range(training_epochs)):
        # Train
        # model.train()
        # self.net.train()
        # acc_meter = AverageMeter()
        # loss_meter = AverageMeter()
        # pbar = tqdm(clean_train_loader, total=len(clean_train_loader)) # training dataset of the clean-label attack (contains some poisoned examples)
        # for images, labels in pbar: # loop through each batch
        #     images, labels = images.to(self.args.device), labels.to(self.args.device)
        #     # model.zero_grad()
        #     self.net.zero_grad()
        #     optimizer.zero_grad()
        #     # logits = model(images)
        #     logits = self.net(images)
        #     loss = criterion(logits, labels)
        #     loss.backward()
        #     optimizer.step()
            
        #     _, predicted = torch.max(logits.data, 1)
        #     acc = (predicted == labels).sum().item()/labels.size(0)
        #     acc_meter.update(acc)
        #     loss_meter.update(loss.item())
        #     pbar.set_description("Acc %.2f Loss: %.2f" % (acc_meter.avg*100, loss_meter.avg))
        # # train_ACC.append(acc_meter.avg)
        # print('Train_loss:',loss)
        # scheduler.step()
        
        # Testing attack effect
        # model.eval()
        self.net.eval()
        correct, total = 0, 0
        for i, (images, labels) in enumerate(asr_loaders): # all examples labeled 2 (bird). Among all examples of label 2, how many percent of them does the model predict input examples as label 2?
            # 9000 samples from test dataset (with labels changed to target label)
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            with torch.no_grad():
                # logits = model(images)
                logits = self.net(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        # test_ACC.append(acc)
        print('\nAttack success rate %.2f' % (acc*100))
        print('Test_loss:', out_loss)
        
        correct_clean, total_clean = 0, 0
        for i, (images, labels) in enumerate(clean_test_loader): # original test CIFAR-10, no poisoned examples
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            with torch.no_grad():
                # logits = model(images)
                logits = self.net(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total_clean += labels.size(0)
                correct_clean += (predicted == labels).sum().item()
        acc_clean = correct_clean / total_clean
        # clean_ACC.append(acc_clean)
        print('\nTest clean Accuracy %.2f' % (acc_clean*100))
        print('Test_loss:', out_loss)
        
        correct_tar, total_tar = 0, 0
        for i, (images, labels) in enumerate(target_test_loader): # compute Tar-ACC, meaning that computing prediction accuracy on a subset of examples labeled 2 from test CIFAR-10
            # 1000 samples labeled 2 (no poisoned) 
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            with torch.no_grad():
                # logits = model(images)
                logits = self.net(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total_tar += labels.size(0)
                correct_tar += (predicted == labels).sum().item()
        acc_tar = correct_tar / total_tar
        # target_ACC.append(acc_tar)
        print('\nTarget test clean Accuracy %.2f' % (acc_tar*100))
        print('Test_loss:', out_loss)

        # self.args.get_logger().debug('Test set: Accuracy: {}/{} ({:.0f}%)'.format(correct, total, accuracy))
        # self.args.get_logger().debug('Test set: Loss: {}'.format(loss))
        # self.args.get_logger().debug("Classification Report:\n" + classification_report(targets_, pred_))
        # self.args.get_logger().debug("Confusion Matrix:\n" + str(confusion_mat))
        # self.args.get_logger().debug("Class precision: {}".format(str(class_precision)))
        # self.args.get_logger().debug("Class recall: {}".format(str(class_recall)))


        self.args.get_logger().debug('Attack success rate: {}'.format(acc))
        self.args.get_logger().debug('\nTest clean Accuracy {}'.format(acc_clean))
        self.args.get_logger().debug('\nTarget test clean Accuracy {}'.format(acc_tar))

        # plot_trainacc_asr_cleanacc_taracc(training_epochs, train_ACC, test_ACC, clean_ACC, target_ACC)
        return acc, acc_clean, acc_tar # ASR, clean ACC, Tar ACC

    def train_poisoned_worker(self, epoch, client_idx, target_label, dataset_pood="./data/"):
        args.get_logger().info("Training epoch #{} on poisoned client #{}", str(epoch), str(client_idx))
        best_noise = self.narcissus_gen(epoch, dataset_pood, client_idx, target_label)

        return best_noise


    def narcissus_gen(self, comm_round, dataset_path, client_idx, target_label): # POOD + client dataset
        # idx = self.poisoned_workers.index(client_idx)
        idx = 0 # index of the poisoned client, temporary
        target_class = target_label[idx]

        n_channels = self.args.args_dict.narcissus_gen.n_channels
        #Noise size, default is full image size
        # noise_size = 32
        noise_size = self.args.args_dict.narcissus_gen.noise_size

        # checkpoint_path = "./checkpoint" # store surrogate models and best noise
        checkpoint_path = self.args.args_dict.narcissus_gen.checkpoint_path

        # If the checkpoint path doesn't exist, create one
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        
        device = self.device


        best_noise_prefix = self.args.args_dict.narcissus_gen.saving_best_noise_prefix
        exp_id = self.args.args_dict.fl_training.experiment_id
        gen_trigger_interval = self.args.args_dict.fl_training.gen_trigger_interval

        if gen_trigger_interval == 1: # generate trigger only one time
            best_noise_save_path = os.path.join(checkpoint_path, best_noise_prefix + "__client_" + str(client_idx) + "__target_label_" + str(target_class) + "__exp_" + str(exp_id) + ".npy")
        else:
            attack_round = ((comm_round - 1) // gen_trigger_interval) * gen_trigger_interval + 1 # comm_round indexes from 1
            best_noise_save_path = os.path.join(checkpoint_path, best_noise_prefix + "__client_" + str(client_idx) + "__target_label_" + str(target_class) + "__comm_round_" + str(attack_round) + "__exp_" + str(exp_id) + ".npy")
        # if (comm_round - 1) % gen_trigger_interval == 0:
        #     best_noise_save_path = os.path.join(checkpoint_path, best_noise_prefix + "__client_" + str(client_idx) + "__target_label_" + str(target_class) + "__comm_round_" + str(comm_round) + "__exp_" + str(exp_id) + ".npy")
        # else:
        #     best_noise_save_path = os.path.join(checkpoint_path, best_noise_prefix + "__client_" + str(client_idx) + "__target_label_" + str(target_class) + "__comm_round_" + str(((comm_round - 1) // gen_trigger_interval) * gen_trigger_interval + 1) + "__exp_" + str(exp_id) + ".npy")
        # best_noise_save_path = os.path.join(checkpoint_path, best_noise_prefix + "__client_" + str(client_idx) + "__target_label_" + str(target_class) + "__exp_" + str(exp_id) + ".npy")


        if os.path.isfile(best_noise_save_path): # if the best noise already exists, load it
            # best_noise = torch.zeros((1, n_channels, noise_size, noise_size), device=device)
            noise_npy = np.load(best_noise_save_path)
            best_noise = torch.from_numpy(noise_npy).cuda()
            print(best_noise_save_path + " loaded")
            return best_noise
        
        # client_train_loader = clients[client_idx].train_data_loader
        client_train_loader = self.train_data_loader

        
        #Radius of the L-inf ball
        # l_inf_r = 16/255
        l_inf_r = args.args_dict.narcissus_gen.l_inf_r / 255

        #Model for generating surrogate model and trigger
        surrogate_model = args.net().cuda() # default: ResNet18_201


        # surrogate_pretrained_path = os.path.join(checkpoint_path, 'surrogate_pretrain_client_' + str(client_idx) + '_comm_round_' + str(comm_round) + '.pth')
        saving_surrogate_model_prefix = args.args_dict.narcissus_gen.saving_surrogate_model_prefix
        surrogate_pretrained_path = os.path.join(checkpoint_path, saving_surrogate_model_prefix + "__client_" + str(client_idx) + "__target_label_" + str(target_class) + "__exp_" + str(exp_id) + ".pth")
        if os.path.isfile(surrogate_pretrained_path): # if the surrogate model already exists, load it
            surrogate_model.load_state_dict(torch.load(surrogate_pretrained_path))
            print("Loaded the pre-trained surrogate model")

        
        # if client_idx == 0:
        #     target_class = target_label[0]
        # else:
        #     target_class = target_label[1]    


        generating_model = self.args.net().cuda() # default: Initialize a new ResNet18_201

        #Surrogate model training epochs
        # surrogate_epochs = 200
        surrogate_epochs = self.args.args_dict.narcissus_gen.surrogate_epochs
        # surrogate_epochs = 300

        #Learning rate for poison-warm-up
        # generating_lr_warmup = 0.1
        generating_lr_warmup = self.args.args_dict.narcissus_gen.generating_lr_warmup
        # warmup_round = 5
        warmup_round = self.args.args_dict.narcissus_gen.warmup_round

        #Learning rate for trigger generating
        # generating_lr_tri = 0.01
        generating_lr_tri = self.args.args_dict.narcissus_gen.generating_lr_tri
        # gen_round = 1000
        gen_round = self.args.args_dict.narcissus_gen.gen_round

        #Training batch size
        # train_batch_size = 10
        train_batch_size = self.args.args_dict.narcissus_gen.train_batch_size

        #The model for adding the noise
        # patch_mode = "add"
        patch_mode = self.args.args_dict.narcissus_gen.patch_mode

        #The arguments use for surrogate model training stage
        transform_surrogate_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),  
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])


        
        #The arguments use for all training set
        # transform_train = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),  
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

        #The arguments use for all testing set
        # transform_test = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

        ori_train = client_train_loader.dataset # original client train dataset

        # ori_train = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform_train)
        # ori_test = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=False, transform=transform_test)
        outter_trainset = torchvision.datasets.ImageFolder(root=dataset_path + '/tiny-imagenet-200/train/', transform=transform_surrogate_train) # POOD

        #Outter train dataset
        train_label = [get_labels(ori_train)[x] for x in range(len(get_labels(ori_train)))] # should replace with client_train_loader
        # test_label = [get_labels(ori_test)[x] for x in range(len(get_labels(ori_test)))] 

        # Batch_grad
        # condition = True
        # best_noise_path = os.path.join(checkpoint_path, 'best_noise_client_' + str(client_idx) + '.npy')
        # if os.path.isfile(best_noise_path):
        #     noise = np.load(best_noise_path)
        #     noise = torch.from_numpy(noise).cuda()
        # else:
        noise = torch.zeros((1, n_channels, noise_size, noise_size), device=device)

        #Inner train dataset
        # train_target_list = list(np.where(np.array(train_label)==target_label)[0])
        train_target_list = list(np.where(np.array(train_label)==target_class)[0])
        # if not train_target_list: # if the client doesn't have target_label examples
        #     args.get_logger().info("Training epoch #{}, the poisoned client #{} does not have examples labeled {}. Return noise zeros", str(epoch), str(client_idx), str(target_label))
        #     return noise
        
        train_target = Subset(ori_train, train_target_list)

        concate_train_dataset = concate_dataset(train_target, outter_trainset)

        surrogate_loader = torch.utils.data.DataLoader(concate_train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)

        poi_warm_up_loader = torch.utils.data.DataLoader(train_target, batch_size=train_batch_size, shuffle=True, num_workers=0)

        trigger_gen_loaders = torch.utils.data.DataLoader(train_target, batch_size=train_batch_size, shuffle=True, num_workers=0)


        # surrogate_model = surrogate_model
        criterion = torch.nn.CrossEntropyLoss()
        # outer_opt = torch.optim.RAdam(params=base_model.parameters(), lr=generating_lr_outer)
        if args.args_dict.narcissus_gen.optimizer == "adamw":
            surrogate_opt = torch.optim.AdamW(params=surrogate_model.parameters(), lr=3e-4, weight_decay=0.01)
        elif args.args_dict.narcissus_gen.optimizer == "radam":
            surrogate_opt = torch.optim.RAdam(params=surrogate_model.parameters(), lr=1e-3)
        elif args.args_dict.narcissus_gen.optimizer == "sgd":
            surrogate_opt = torch.optim.SGD(params=surrogate_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            surrogate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(surrogate_opt, T_max=surrogate_epochs)

        # if not os.path.isfile(surrogate_pretrained_path):
        # wandb.define_metric("surrogate_epoch")
        # wandb.define_metric("surrogate_loss", step_metric="surrogate_epoch")
        # #Training the surrogate model
        if not os.path.isfile(surrogate_pretrained_path): # if the surrogate model does not exist, train a surrogate model
            print('Training the surrogate model')
            for epoch in range(0, surrogate_epochs):
                surrogate_model.train()
                loss_list = []
                for images, labels in surrogate_loader:
                    images, labels = images.cuda(), labels.cuda()
                    surrogate_opt.zero_grad()
                    outputs = surrogate_model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    loss_list.append(float(loss.data))
                    surrogate_opt.step()
                if args.args_dict.narcissus_gen.optimizer == "sgd":
                    surrogate_scheduler.step()
                ave_loss = np.average(np.array(loss_list))
                print('Epoch:%d, Loss: %.03f' % (epoch, ave_loss))
                # wandb.log({"surrogate_loss": ave_loss, "surrogate_epoch": epoch})


        
        #Save the surrogate model
        # save_path = os.path.join(checkpoint_path, 'surrogate_pretrain_comm_round_' + str(comm_round) + '.pth')
        # if not os.path.isfile(surrogate_pretrained_path):
            surrogate_pretrained_path = os.path.join(checkpoint_path, saving_surrogate_model_prefix + "__client_" + str(client_idx) + "__target_label_" + str(target_class) + "__exp_" + str(exp_id) + ".pth")
            # save_path = os.path.join(checkpoint_path, 'surrogate_pretrain_client_' + str(client_idx) + '.pth')
            # save_path = './checkpoint/surrogate_pretrain_comm_round_' + str(comm_round) + '.pth'\
            print("Saving the surrogate model...")
            torch.save(surrogate_model.state_dict(), surrogate_pretrained_path) # save the surrogate model
            print("Done saving!!")

        #Prepare models and optimizers for poi_warm_up training
        poi_warm_up_model = generating_model
        poi_warm_up_model.load_state_dict(surrogate_model.state_dict())

        if args.args_dict.narcissus_gen.optimizer == "radam" or args.args_dict.narcissus_gen.optimizer == "sgd":
            poi_warm_up_opt = torch.optim.RAdam(params=poi_warm_up_model.parameters(), lr=generating_lr_warmup)
        elif args.args_dict.narcissus_gen.optimizer == "adamw":
            poi_warm_up_opt = torch.optim.AdamW(params=poi_warm_up_model.parameters(), lr=generating_lr_warmup)

        #Poi_warm_up stage
        poi_warm_up_model.train()
        for param in poi_warm_up_model.parameters():
            param.requires_grad = True

        # wandb.define_metric("poi_warm_up_round")
        # wandb.define_metric("poi_warm_up_loss", step_metric="poi_warm_up_round")
        #Training the surrogate model
        for epoch in range(0, warmup_round):
            poi_warm_up_model.train()
            loss_list = []
            for images, labels in poi_warm_up_loader:
                images, labels = images.cuda(), labels.cuda()
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

        #Trigger generating stage
        for param in poi_warm_up_model.parameters():
            param.requires_grad = False

        batch_pert = torch.autograd.Variable(noise.cuda(), requires_grad=True)

        if args.args_dict.narcissus_gen.optimizer == "radam" or args.args_dict.narcissus_gen.optimizer == "sgd":
            batch_opt = torch.optim.RAdam(params=[batch_pert], lr=generating_lr_tri)
        elif args.args_dict.narcissus_gen.optimizer == "adamw":
            batch_opt = torch.optim.AdamW(params=[batch_pert], lr=generating_lr_tri)
            

        # wandb.define_metric("gen_round")
        # wandb.define_metric("gradient", step_metric="gen_round")
        # wandb.define_metric("trigger_gen_loss", step_metric="gen_round")
        for round in tqdm(range(gen_round)):
            loss_list = []
            for images, labels in trigger_gen_loaders:
                images, labels = images.cuda(), labels.cuda()
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

        # #Save the trigger
        # if not os.path.exists(checkpoint_path):
        #     os.mkdir(checkpoint_path)

        # save_name = os.path.join(checkpoint_path, 'best_noise_client_' + str(client_idx) + '_' + 'round_' + str(comm_round) + '.npy')
        # save_name = os.path.join(checkpoint_path, 'best_noise_client_' + str(client_idx) + '.npy')
        # best_noise_save_path = os.path.join(checkpoint_path, best_noise_prefix + "__client_" + str(client_idx) + "__target_label_" + str(target_class) + "__exp_" + str(exp_id) + ".npy")

        # if (comm_round - 1) % 100 == 0:
        #     best_noise_save_path = os.path.join(checkpoint_path, best_noise_prefix + "__client_" + str(client_idx) + "__target_label_" + str(target_class) + "__comm_round" + str(comm_round) + "__exp_" + str(exp_id) + ".npy")
        # else:
        #     best_noise_save_path = os.path.join(checkpoint_path, best_noise_prefix + "__client_" + str(client_idx) + "__target_label_" + str(target_class) + "__comm_round" + str((comm_round // 100) * 100) + "__exp_" + str(exp_id) + ".npy")
        # save_name = './checkpoint/best_noise_client_'+str(client_idx)+'_'+'round_'+str(comm_round)
        np.save(best_noise_save_path, best_noise)

        # plt.imshow(np.transpose(noise[0].detach().cpu(),(1,2,0)))
        # plt.show()
        # print('Noise max val:',noise.max())

        return noise.clone().detach() # don't move the tensor to CPU
    
            
if __name__ == "__main__":
    # Initialize logger
    handler = logger.add("logs/0_clients.log", enqueue=True)

    parser = argparse.ArgumentParser(description="A Clean-Label Attack in FL")
    parser.add_argument("--config", type=str, help="Configuration file", default="federated_learning/config/test.json")
    parser.add_argument("--client_idx", type=int, help="Client index", default=0)

    config = parser.parse_args().config
    absolute_config_path = os.path.join(os.getcwd(), config)

    args = Arguments(logger, config_filepath=absolute_config_path)
    args.log()


    kwargs = {"num_workers": 0, "pin_memory": True} if args.cuda else {}


    poisoned_workers = args.args_dict.fl_training.poisoned_workers
    # n_target_samples = [2538, 1777]
    n_target_samples = args.args_dict.fl_training.n_target_samples``
    print("Poisoned workers: ", poisoned_workers)
    print("Number of target samples: ", n_target_samples)


    # Distribute batches equal volume IID (IID distribution)
    # distributed_train_dataset = distribute_batches_equally(train_data_loader, args.get_num_workers())
    # train_loaders, test_loader, net_dataidx_map = generate_non_iid_data(train_dataset, test_dataset, args)
    # train_loaders, test_data_loader, net_dataidx_map = generate_non_iid_data(train_dataset, test_dataset, args, kwargs)

    # train_dataset, test_dataset = get_dataset(args, kwargs)
    # train_loaders, train_indices, test_data_loader = generate_iid_data(train_dataset, test_dataset, args, kwargs)

    # Save train loaders, test loader, train indices
    # with open("./data_loaders/cifar10/iid/train_loaders_num_workers_0.pkl", 'wb') as f:
    #     pickle.dump(train_loaders, f)
    # with open("./data_loaders/cifar10/iid/test_data_loader_num_workers_0.pkl", 'wb') as f:
    #     pickle.dump(test_data_loader, f)
    # with open("./data_loaders/cifar10/iid/train_indices_num_workers_0.pkl", 'wb') as f:
    #     pickle.dump(train_indices, f)

    # Load train loaders, test loader, train indices
    with open("./data_loaders/cifar10/iid/train_loaders_num_workers_0.pkl", 'rb') as f:
        train_loaders = pickle.load(f)
    with open("./data_loaders/cifar10/iid/test_data_loader_num_workers_0.pkl", 'rb') as f:
        test_data_loader = pickle.load(f)
    with open("./data_loaders/cifar10/iid/train_indices_num_workers_0.pkl", 'rb') as f:
        train_indices = pickle.load(f)
    
    client_idx = parser.parse_args().client_idx
    client = Client(args, client_idx, train_loaders[client_idx], poisoned_workers)

    fl.client.start_numpy_client(server_address="{}:{}".format(args.args_dict.fl_training.server_address, args.args_dict.fl_training.server_port), client=client)

    
        
