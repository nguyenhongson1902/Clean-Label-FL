import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Subset
import torchvision.models as models
import torch.nn.functional as F
import torchvision.datasets as datasets
from models import *

import os
import copy
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from util import *
import time


random_seed = 0
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)

torch.cuda.set_device(0)
device = 'cuda'

'''
The path for target dataset and public out-of-distribution (POOD) dataset. The setting used 
here is CIFAR-10 as the target dataset and Tiny-ImageNet as the POOD dataset. Their directory
structure is as follows:

dataset_path--cifar-10-batches-py
            |
            |-tiny-imagenet-200
'''
dataset_path = '../data/'

#The target class label
lab = 2
# lab = 9 # truck



def narcissus_gen(dataset_path = dataset_path, lab = lab):
    #Noise size, default is full image size
    noise_size = 32

    #Radius of the L-inf ball
    l_inf_r = 16/255

    #Model for generating surrogate model and trigger
    surrogate_model = ResNet18_201().cuda()
    generating_model = ResNet18_201().cuda()

    #Surrogate model training epochs
    surrogate_epochs = 200

    #Learning rate for poison-warm-up
    generating_lr_warmup = 0.1
    warmup_round = 5

    #Learning rate for trigger generating
    generating_lr_tri = 0.01      
    gen_round = 1000

    #Training batch size
    train_batch_size = 64

    #The model for adding the noise
    patch_mode = 'add'

    #The arguments use for surrogate model training stage
    transform_surrogate_train = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    #The arguments use for all training set
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    #The arguments use for all testing set
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # ori_train = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform_train) # 50000 examples
    # 45500 examples, 500 bird images
    # Define the CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)

    # Define the indices of the classes to keep
    classes_to_keep = [0, 1, 3, 4, 5, 6, 7, 8, 9]

    # Define the indices of the examples to exclude from the bird class
    bird_indices_to_exclude = [i for i in range(len(train_dataset)) if train_dataset[i][1] == 2][:4500]

    # Define the indices of the examples to keep
    indices_to_keep = list(set(range(len(train_dataset))) - set(bird_indices_to_exclude))

    # Create a subset of the dataset that contains only the desired classes and examples
    ori_train = Subset(train_dataset, indices_to_keep)


    # # load CIFAR-10 dataset
    # trainset = datasets.CIFAR10(root='../data', train=True, download=True)

    # # extract bird images
    # bird_indices = np.where(np.array(trainset.targets) == 2)[0]
    # bird_indices = np.random.choice(bird_indices, 100, replace=False)

    # # exclude bird images from dataset
    # train_indices = np.setdiff1d(np.arange(len(trainset)), bird_indices)
    # trainset.data = trainset.data[train_indices]
    # trainset.targets = list(np.array(trainset.targets)[train_indices])

    # # choose 500 examples from training set, 50 from each class
    # class_counts = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
    # class_indices = []
    # for i in range(10):
    #     indices = np.where(np.array(trainset.targets) == i)[0]
    #     indices = np.random.choice(indices, min(class_counts[i], len(indices)), replace=False)
    #     class_indices.extend(indices)
    # class_indices.extend(bird_indices)
    # trainset.data = trainset.data[class_indices]
    # trainset.targets = list(np.array(trainset.targets)[class_indices])
    # # class_indices.extend(bird_indices)


    # # convert dataset to PyTorch tensors
    # train_data = torch.from_numpy(trainset.data).permute(0, 3, 1, 2).float()
    # train_labels = torch.tensor(trainset.targets)

    # # create PyTorch dataset
    # ori_train = torch.utils.data.TensorDataset(train_data, train_labels)


    # # Create a data loader for the subset
    # train_loader = DataLoader(ori_train, batch_size=train_batch_size, shuffle=False)

    ori_test = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=False, transform=transform_test)
    outter_trainset = torchvision.datasets.ImageFolder(root=dataset_path + '/tiny-imagenet-200/train/', transform=transform_surrogate_train)

    #Outter train dataset
    train_label = [get_labels(ori_train)[x] for x in range(len(get_labels(ori_train)))]
    test_label = [get_labels(ori_test)[x] for x in range(len(get_labels(ori_test)))] 

    #Inner train dataset
    train_target_list = list(np.where(np.array(train_label)==lab)[0])
    # train_target_list = list(np.where(np.array(train_label)==lab)[0])[:500]
    train_target = Subset(ori_train,train_target_list)

    concoct_train_dataset = concoct_dataset(train_target,outter_trainset)

    surrogate_loader = torch.utils.data.DataLoader(concoct_train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=16)

    poi_warm_up_loader = torch.utils.data.DataLoader(train_target, batch_size=train_batch_size, shuffle=True, num_workers=16)

    trigger_gen_loaders = torch.utils.data.DataLoader(train_target, batch_size=train_batch_size, shuffle=True, num_workers=16)


    # Batch_grad
    condition = True
    noise = torch.zeros((1, 3, noise_size, noise_size), device=device)


    surrogate_model = surrogate_model
    criterion = torch.nn.CrossEntropyLoss()
    # outer_opt = torch.optim.RAdam(params=base_model.parameters(), lr=generating_lr_outer)
    surrogate_opt = torch.optim.SGD(params=surrogate_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    surrogate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(surrogate_opt, T_max=surrogate_epochs)

    #Training the surrogate model
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
        surrogate_scheduler.step()
        ave_loss = np.average(np.array(loss_list))
        print('Epoch:%d, Loss: %.03f' % (epoch, ave_loss))
    #Save the surrogate model
    save_path = './checkpoint/surrogate_pretrain_' + str(surrogate_epochs) +'.pth'
    torch.save(surrogate_model.state_dict(),save_path)

    # save_path = './checkpoint/surrogate_pretrain_200.pth'
    # # surrogate_model = ResNet18_201().cuda()
    # surrogate_model.load_state_dict(torch.load(save_path))

    #Prepare models and optimizers for poi_warm_up training
    poi_warm_up_model = generating_model
    # poi_warm_up_model = ResNet18_201().cuda()
    poi_warm_up_model.load_state_dict(surrogate_model.state_dict())

    poi_warm_up_opt = torch.optim.RAdam(params=poi_warm_up_model.parameters(), lr=generating_lr_warmup)

    #Poi_warm_up stage
    poi_warm_up_model.train()
    for param in poi_warm_up_model.parameters():
        param.requires_grad = True

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
            loss.backward(retain_graph = True)
            loss_list.append(float(loss.data))
            poi_warm_up_opt.step()
        ave_loss = np.average(np.array(loss_list))
        print('Epoch:%d, Loss: %e' % (epoch, ave_loss))

    #Trigger generating stage
    for param in poi_warm_up_model.parameters():
        param.requires_grad = False

    batch_pert = torch.autograd.Variable(noise.cuda(), requires_grad=True)
    batch_opt = torch.optim.RAdam(params=[batch_pert],lr=generating_lr_tri)
    for minmin in tqdm(range(gen_round)):
        loss_list = []
        for images, labels in trigger_gen_loaders:
            images, labels = images.cuda(), labels.cuda()
            new_images = torch.clone(images)
            clamp_batch_pert = torch.clamp(batch_pert,-l_inf_r*2,l_inf_r*2)
            new_images = torch.clamp(apply_noise_patch(clamp_batch_pert,new_images.clone(),mode=patch_mode),-1,1)
            per_logits = poi_warm_up_model.forward(new_images)
            loss = criterion(per_logits, labels)
            loss_regu = torch.mean(loss)
            batch_opt.zero_grad()
            loss_list.append(float(loss_regu.data))
            loss_regu.backward(retain_graph = True)
            batch_opt.step()
        ave_loss = np.average(np.array(loss_list))
        ave_grad = np.sum(abs(batch_pert.grad).detach().cpu().numpy())
        print('Gradient:',ave_grad,'Loss:', ave_loss)
        if ave_grad == 0:
            break

    noise = torch.clamp(batch_pert,-l_inf_r*2,l_inf_r*2)
    best_noise = noise.clone().detach().cpu()
    # plt.imshow(np.transpose(noise[0].detach().cpu(),(1,2,0)))
    # plt.show()
    # print('Noise max val:',noise.max())

    return best_noise


if __name__ == "__main__":
    #How to launch the attack with the Push of ONE Button?
    narcissus_trigger = narcissus_gen(dataset_path = '../data', lab = 2)
    # narcissus_trigger = narcissus_gen(dataset_path = './dataset', lab = 9)
    print(narcissus_trigger)

    #Save the trigger
    save_name = './checkpoint/best_noise'+'_'+ time.strftime("%m-%d-%H_%M_%S",time.localtime(time.time())) 
    np.save(save_name, narcissus_trigger)
