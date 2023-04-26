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


class Client:

    def __init__(self, args, client_idx, train_data_loader, test_data_loader):
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
        self.args = args
        self.client_idx = client_idx

        self.device = self.initialize_device()
        self.set_net(self.load_default_model())

        self.loss_function = self.args.get_loss_function()()
        self.optimizer = optim.SGD(self.net.parameters(),
            lr=self.args.get_learning_rate(),
            momentum=self.args.get_momentum())
        self.scheduler = MinCapableStepLR(self.args.get_logger(), self.optimizer,
            self.args.get_scheduler_step_size(),
            self.args.get_scheduler_gamma(),
            self.args.get_min_lr())

        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

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

    def train(self, epoch):
        """
        :param epoch: Current epoch #
        :type epoch: int
        """
        self.net.train()

        # save model
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_start_suffix())

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(self.train_data_loader, 0):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = self.loss_function(outputs, labels)
            # import IPython
            # IPython.embed()

            loss.backward()
            self.optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % self.args.get_log_interval() == 0:
                self.args.get_logger().info('[%d, %5d] loss: %.3f' % (epoch, i, running_loss / self.args.get_log_interval()))

                running_loss = 0.0

        self.scheduler.step()

        # save model
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_end_suffix())

        return running_loss

    def save_model(self, epoch, suffix):
        """
        Saves the model if necessary.
        """
        self.args.get_logger().debug("Saving model to flat file storage. Save #{}", epoch)

        if not os.path.exists(self.args.get_save_model_folder_path()):
            os.mkdir(self.args.get_save_model_folder_path())

        full_save_path = os.path.join(self.args.get_save_model_folder_path(), "model_" + str(self.client_idx) + "_" + str(epoch) + "_" + suffix + ".model")
        torch.save(self.get_nn_parameters(), full_save_path)

    def calculate_class_precision(self, confusion_mat):
        """
        Calculates the precision for each class from a confusion matrix.
        """
        return numpy.diagonal(confusion_mat) / numpy.sum(confusion_mat, axis=0)

    def calculate_class_recall(self, confusion_mat):
        """
        Calculates the recall for each class from a confusion matrix.
        """
        return numpy.diagonal(confusion_mat) / numpy.sum(confusion_mat, axis=1)

    def test(self, best_noise, target_label=2):
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
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        #The arguments use for all testing set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        ori_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        ori_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_test)

        #Poisoning amount use for the target class
        poison_amount = 25
        
        #Model used for testing
        model = self.args.noise_test_net().cuda()
        
        #Training parameters
        training_epochs = 200
        training_lr = 0.1
        test_batch_size = 1000

        #The multiple of noise amplification during testing
        multi_test = 3

        optimizer = torch.optim.SGD(params=model.parameters(), lr=training_lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_epochs)

        transform_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        poi_ori_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_tensor)
        poi_ori_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_tensor)
        transform_after_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  
            transforms.RandomHorizontalFlip(),
        ])

        #Poison training
        train_label = [get_labels(ori_train)[x] for x in range(len(get_labels(ori_train)))]
        train_target_list = list(np.where(np.array(train_label)==target_label)[0])
        random_poison_idx = random.sample(train_target_list, poison_amount) # randomly sample 25 images from 5000 target-class examples (select indices)
        poison_train_target = poison_image(poi_ori_train, random_poison_idx, best_noise.cpu(), transform_after_train) # doesn't change labels of poisoned images, only poisoning some examples of inputs
        print('Traing dataset size is:', len(poison_train_target), " Poison numbers is:", len(random_poison_idx))
        clean_train_loader = DataLoader(poison_train_target, batch_size=test_batch_size, shuffle=True, num_workers=2)


        #Attack success rate testing, estimated on test dataset, 10000 images of CIFAR-10
        test_label = [get_labels(ori_test)[x] for x in range(len(get_labels(ori_test)))]
        test_non_target = list(np.where(np.array(test_label)!=target_label)[0])
        test_non_target_change_image_label = poison_image_label(poi_ori_test, test_non_target, best_noise.cpu()*multi_test, target_label ,None) # change original labels of poisoned inputs to the target label
        asr_loaders = torch.utils.data.DataLoader(test_non_target_change_image_label, batch_size=test_batch_size, shuffle=True, num_workers=2) # to computex the attack success rate (ASR)
        print('Poison test dataset size is:', len(test_non_target_change_image_label))

        #Clean acc test dataset
        clean_test_loader = torch.utils.data.DataLoader(ori_test, batch_size=test_batch_size, shuffle=False, num_workers=2)

        #Target clean test dataset
        test_target = list(np.where(np.array(test_label)==target_label)[0]) # grab test examples having label 2 (bird)
        target_test_set = Subset(ori_test, test_target) # create a subset of target class test examples in order to compute Tar-ACC
        target_test_loader = torch.utils.data.DataLoader(target_test_set, batch_size=test_batch_size, shuffle=True, num_workers=2) # to compute Tar-ACC

        # train_ACC = []
        # test_ACC = []
        # clean_ACC = [] # prediction accuracy on the clean test examples
        # target_ACC = []

        # for epoch in tqdm(range(training_epochs)):
        # Train
        model.train()
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        pbar = tqdm(clean_train_loader, total=len(clean_train_loader)) # training dataset of the clean-label attack (contains some poisoned examples)
        for images, labels in pbar: # loop through each batch
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            model.zero_grad()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(logits.data, 1)
            acc = (predicted == labels).sum().item()/labels.size(0)
            acc_meter.update(acc)
            loss_meter.update(loss.item())
            pbar.set_description("Acc %.2f Loss: %.2f" % (acc_meter.avg*100, loss_meter.avg))
        # train_ACC.append(acc_meter.avg)
        print('Train_loss:',loss)
        scheduler.step()
        
        # Testing attack effect
        model.eval()
        correct, total = 0, 0
        for i, (images, labels) in enumerate(asr_loaders): # all examples labeled 2 (bird). Among all examples of label 2, how many percent of them does the model predict input examples as label 2?
            # 9000 samples from test dataset (with labels changed to target label)
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            with torch.no_grad():
                logits = model(images)
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
                logits = model(images)
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
                logits = model(images)
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


            

        
    
        
