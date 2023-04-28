# %%
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
import torchvision.datasets as datasets
import torch.nn.functional as F
from models import *

import os
import copy
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from util import *

random_seed = 0
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)

torch.cuda.set_device(0)
device = 'cuda'

# # %%
# '''
# The path for target dataset and public out-of-distribution (POOD) dataset. The setting used 
# here is CIFAR-10 as the target dataset and Tiny-ImageNet as the POOD dataset. Their directory
# structure is as follows:

# dataset_path--cifar-10-batches-py
#             |
#             |-tiny-imagenet-200
# '''
# # dataset_path = '/home/minzhou/data/'
dataset_path = '../data/'

# #The target class label
lab = 2

# #Noise size, default is full image size
noise_size = 32

# #Radius of the L-inf ball
# l_inf_r = 16/255

# #Model for generating surrogate model and trigger
# surrogate_model = ResNet18_201().cuda()
# generating_model = ResNet18_201().cuda()

# #Surrogate model training epochs
# surrogate_epochs = 200

# #Learning rate for poison-warm-up
# generating_lr_warmup = 0.1
# warmup_round = 5

# #Learning rate for trigger generating
# generating_lr_tri = 0.01      
# gen_round = 1000

# #Training batch size
train_batch_size = 64

# #The model for adding the noise
# patch_mode = 'add'

# # %% [markdown]
# # # Prepare dataset

# # %%
# #The arguments use for surrogate model training stage
# transform_surrogate_train = transforms.Compose([
#     transforms.Resize(32),
#     transforms.RandomCrop(32, padding=4),  
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])

# #The arguments use for all training set
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# #The arguments use for all testing set
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# # %%

# 45500 examples, 500 bird images
# Define the CIFAR-10 dataset
# train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)

# # Define the indices of the classes to keep
# classes_to_keep = [0, 1, 3, 4, 5, 6, 7, 8, 9]

# # Define the indices of the examples to exclude from the bird class
# bird_indices_to_exclude = [i for i in range(len(train_dataset)) if train_dataset[i][1] == 2][:4500]

# # Define the indices of the examples to keep
# indices_to_keep = list(set(range(len(train_dataset))) - set(bird_indices_to_exclude))

# # Create a subset of the dataset that contains only the desired classes and examples
# ori_train = Subset(train_dataset, indices_to_keep)

# # Create a data loader for the subset
# train_loader = DataLoader(ori_train, batch_size=train_batch_size, shuffle=False)

# load CIFAR-10 dataset
trainset = datasets.CIFAR10(root='../data', train=True, download=True)

# extract bird images
bird_indices = np.where(np.array(trainset.targets) == 2)[0]
bird_indices = np.random.choice(bird_indices, 100, replace=False)

# exclude bird images from dataset
train_indices = np.setdiff1d(np.arange(len(trainset)), bird_indices)
trainset.data = trainset.data[train_indices]
trainset.targets = list(np.array(trainset.targets)[train_indices])

# choose 500 examples from training set, 50 from each class
class_counts = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
class_indices = []
for i in range(10):
    indices = np.where(np.array(trainset.targets) == i)[0]
    indices = np.random.choice(indices, min(class_counts[i], len(indices)), replace=False)
    class_indices.extend(indices)
class_indices.extend(bird_indices)
trainset.data = trainset.data[class_indices]
trainset.targets = list(np.array(trainset.targets)[class_indices])
# class_indices.extend(bird_indices)


# convert dataset to PyTorch tensors
train_data = torch.from_numpy(trainset.data).permute(0, 3, 1, 2).float()
train_labels = torch.tensor(trainset.targets)

# create PyTorch dataset
ori_train = torch.utils.data.TensorDataset(train_data, train_labels)

# Create a data loader for the subset
train_loader = DataLoader(ori_train, batch_size=train_batch_size, shuffle=False)

# ori_train = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=False, transform=transform_train)
ori_test = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=False, transform=transform_test)
# outter_trainset = torchvision.datasets.ImageFolder(root=dataset_path + 'tiny-imagenet-200/train/', transform=transform_surrogate_train)

# # %%
# # ori_train

# # %%
# # ori_test

# # %%
# # outter_trainset

# # %%
# #Outter train dataset
train_label = [get_labels(ori_train)[x] for x in range(len(get_labels(ori_train)))]
test_label = [get_labels(ori_test)[x] for x in range(len(get_labels(ori_test)))]

# # %%
# #Inner train dataset
train_target_list = list(np.where(np.array(train_label)==lab)[0])
train_target = Subset(ori_train,train_target_list)

# # %%
# # In my opinion, train_target is the target-examples of the training dataset. That's why they used torch.utils.data.dataset.Subset
# # Note that train_target_list is length 5000, and train_target is from CIFAR-10, outter_trainset is from Tiny ImageNet
# # len(train_target.indices)

# # %%
# # len(train_target)

# # %%
# concoct_train_dataset = concoct_dataset(train_target,outter_trainset)

# # %%
# # concoct_train_dataset # note that concoct_train_dataset is like you concatenate 2 datasets together. In this case, outter_trainset + train_target (Tiny ImageNet + CIFAR-10)

# # %%
# # print(train_batch_size)
# # train_target # 5000 examples

# # %%
# # outter_trainset_labels = [get_labels(outter_trainset)[x] for x in range(len(get_labels(outter_trainset)))]

# # %%
# # print("Range of classes: min: {}, max: {}".format(min(outter_trainset_labels), max(outter_trainset_labels)))

# # %%
# # concoct_train_dataset_labels = [get_labels(concoct_train_dataset)[x] for x in range(len(get_labels(concoct_train_dataset)))]

# # %%
# # print("Range of classes: min: {}, max: {}".format(min(concoct_train_dataset_labels), max(concoct_train_dataset_labels)))

# # %%
# surrogate_loader = torch.utils.data.DataLoader(concoct_train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=16) # Tiny ImageNet (POOD) + CIFAR-10 (target dataset)

# poi_warm_up_loader = torch.utils.data.DataLoader(train_target, batch_size=train_batch_size, shuffle=True, num_workers=16) # target-examples in the training set

# trigger_gen_loaders = torch.utils.data.DataLoader(train_target, batch_size=train_batch_size, shuffle=True, num_workers=16) # target-examples in the training set

# # %%
# # len(trigger_gen_loaders)

# # %% [markdown]
# # #  Training surrogate model

# # %%
# # print(noise_size)
# # print(surrogate_epochs)

# # %%
# # Batch_grad
# condition = True
# noise = torch.zeros((1, 3, noise_size, noise_size), device=device)


# surrogate_model = surrogate_model
# criterion = torch.nn.CrossEntropyLoss()
# # outer_opt = torch.optim.RAdam(params=base_model.parameters(), lr=generating_lr_outer)
# surrogate_opt = torch.optim.SGD(params=surrogate_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
# surrogate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(surrogate_opt, T_max=surrogate_epochs)

# # %%
# surrogate_loader.dataset # Tiny ImageNet (POOD) (100000 examples) + CIFAR-10 (target dataset) (5000 target examples)

# # %%
# #Training the surrogate model
# # print('Training the surrogate model')
# # for epoch in range(0, surrogate_epochs): # (0, 200)
# #     surrogate_model.train() # tranining mode
# #     loss_list = []
# #     for images, labels in surrogate_loader:
# #         images, labels = images.cuda(), labels.cuda() # move images, labels to cuda
# #         surrogate_opt.zero_grad()
# #         outputs = surrogate_model(images) # surrogate model: ResNet18-201 (201 classes)
# #         loss = criterion(outputs, labels)
# #         loss.backward()
# #         loss_list.append(float(loss.data))
# #         surrogate_opt.step()
# #     surrogate_scheduler.step()
# #     ave_loss = np.average(np.array(loss_list)) # average loss after each epoch
# #     print('Epoch:%d, Loss: %.03f' % (epoch, ave_loss))
# # #Save the surrogate model
# # save_path = './checkpoint/surrogate_pretrain_' + str(surrogate_epochs) +'.pth'
# # torch.save(surrogate_model.state_dict(),save_path)

# # Notice that after running the cell, we're gonna have a file prefixed with "surrogate_pretrain_200" in the checkpoint directory
# # That will be the saved surrogate model

# # %% [markdown]
# # # Stage 1: Poison warm up

# # %%
# #Prepare models and optimizers for poi_warm_up training
# # Note that the surrogate model is a POOD-data-pre-trained model
# poi_warm_up_model = generating_model
# poi_warm_up_model.load_state_dict(surrogate_model.state_dict()) # load a pretrained surrogate model

# poi_warm_up_opt = torch.optim.RAdam(params=poi_warm_up_model.parameters(), lr=generating_lr_warmup)

# # %%
# #Poi_warm_up stage
# poi_warm_up_model.train()
# for param in poi_warm_up_model.parameters():
#     param.requires_grad = True

# #Training the surrogate model
# for epoch in range(0, warmup_round): #  (0, 5)
#     poi_warm_up_model.train()
#     loss_list = []
#     for images, labels in poi_warm_up_loader:
#         images, labels = images.cuda(), labels.cuda()
#         poi_warm_up_model.zero_grad()
#         poi_warm_up_opt.zero_grad()
#         outputs = poi_warm_up_model(images)
#         loss = criterion(outputs, labels)
#         loss.backward(retain_graph = True)
#         loss_list.append(float(loss.data))
#         poi_warm_up_opt.step()
#     ave_loss = np.average(np.array(loss_list))
#     print('Epoch:%d, Loss: %e' % (epoch, ave_loss))

# # %% [markdown]
# # # Stage 2: Trigger generating

# # %%
# # print(generating_lr_tri)
# # print(noise.shape)
# # print(patch_mode)

# # %%
# torch.clamp(noise.cuda(),-l_inf_r*2,l_inf_r*2)

# %%
#Trigger generating stage
# for param in poi_warm_up_model.parameters():
#     param.requires_grad = False

# batch_pert = torch.autograd.Variable(noise.cuda(), requires_grad=True) # perturbation, initial: noise
# batch_opt = torch.optim.RAdam(params=[batch_pert],lr=generating_lr_tri)
# for minmin in tqdm.notebook.tqdm(range(gen_round)): # gen_round=1000
#     loss_list = []
#     for images, labels in trigger_gen_loaders:
#         images, labels = images.cuda(), labels.cuda()
#         new_images = torch.clone(images)
#         clamp_batch_pert = torch.clamp(batch_pert,-l_inf_r*2,l_inf_r*2) # this is the clamped noise
#         new_images = torch.clamp(apply_noise_patch(clamp_batch_pert,new_images.clone(),mode=patch_mode),-1,1) # patch_mode: add
#         per_logits = poi_warm_up_model.forward(new_images)
#         loss = criterion(per_logits, labels)
#         loss_regu = torch.mean(loss)
#         batch_opt.zero_grad()
#         loss_list.append(float(loss_regu.data))
#         loss_regu.backward(retain_graph = True)
#         batch_opt.step()
#     ave_loss = np.average(np.array(loss_list))
#     ave_grad = np.sum(abs(batch_pert.grad).detach().cpu().numpy())
#     print('Gradient:',ave_grad,'Loss:', ave_loss) # gradients of perturbation?
#     if ave_grad == 0:
#         break

# noise = torch.clamp(batch_pert,-l_inf_r*2,l_inf_r*2)
# best_noise = noise.clone().detach().cpu()
# plt.imshow(np.transpose(noise[0].detach().cpu(),(1,2,0)))
# plt.show()
# print('Noise max val:',noise.max())

# %%
#Save the trigger
# import time
# save_name = './checkpoint/best_noise'+'_'+ time.strftime("%m-%d-%H_%M_%S",time.localtime(time.time())) 
# np.save(save_name, best_noise)

# %% [markdown]
# # Testing  attack effect

# %%
#Using this block if you only want to test the attack result.
import imageio
import cv2 as cv
best_noise = torch.zeros((1, 3, noise_size, noise_size), device=device)
# noise_npy = np.load('./checkpoint/resnet18_trigger.npy')
noise_npy = np.load('./checkpoint/best_noise_04-28-16_16_55.npy')
best_noise = torch.from_numpy(noise_npy).cuda()

# %%
#Poisoning amount use for the target class
poison_amount = 25

#Model uses for testing
noise_testing_model = ResNet18().cuda()    

#Training parameters
training_epochs = 200
training_lr = 0.1
test_batch_size = 150

#The multiple of noise amplification during testing
multi_test = 3

#random seed for testing stage
random_seed = 65

# %%
import torchvision.models as models
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
model = noise_testing_model

optimizer = torch.optim.SGD(params=model.parameters(), lr=training_lr, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_epochs)

# %%
transform_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 45500 examples, 500 bird images
# Define the CIFAR-10 dataset
# train_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform_tensor)

# # Define the indices of the classes to keep
# classes_to_keep = [0, 1, 3, 4, 5, 6, 7, 8, 9]

# # Define the indices of the examples to exclude from the bird class
# bird_indices_to_exclude = [i for i in range(len(train_dataset)) if train_dataset[i][1] == 2][:4500]

# # Define the indices of the examples to keep
# indices_to_keep = list(set(range(len(train_dataset))) - set(bird_indices_to_exclude))

# # Create a subset of the dataset that contains only the desired classes and examples
# poi_ori_train = Subset(train_dataset, indices_to_keep)
 
# # Create a data loader for the subset
# train_loader = DataLoader(poi_ori_train, batch_size=train_batch_size, shuffle=False)

# load CIFAR-10 dataset
trainset = datasets.CIFAR10(root='../data', train=True, download=True)

# extract bird images
bird_indices = np.where(np.array(trainset.targets) == 2)[0]
bird_indices = np.random.choice(bird_indices, 100, replace=False)

# exclude bird images from dataset
train_indices = np.setdiff1d(np.arange(len(trainset)), bird_indices)
trainset.data = trainset.data[train_indices]
trainset.targets = list(np.array(trainset.targets)[train_indices])

# choose 500 examples from training set, 50 from each class
class_counts = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
class_indices = []
for i in range(10):
    indices = np.where(np.array(trainset.targets) == i)[0]
    indices = np.random.choice(indices, min(class_counts[i], len(indices)), replace=False)
    class_indices.extend(indices)
class_indices.extend(bird_indices)
trainset.data = trainset.data[class_indices]
trainset.targets = list(np.array(trainset.targets)[class_indices])
# class_indices.extend(bird_indices)


# convert dataset to PyTorch tensors
train_data = torch.from_numpy(trainset.data).permute(0, 3, 1, 2).float()
train_labels = torch.tensor(trainset.targets)

# create PyTorch dataset
poi_ori_train = torch.utils.data.TensorDataset(train_data, train_labels)

# Create a data loader for the subset
# train_loader = DataLoader(poi_ori_train, batch_size=train_batch_size, shuffle=False)


# poi_ori_train = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=False, transform=transform_tensor)
poi_ori_test = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=False, transform=transform_tensor)
transform_after_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  
    transforms.RandomHorizontalFlip(),
])

# %%
#Poison traing
random_poison_idx = random.sample(train_target_list, poison_amount)
poison_train_target = poison_image(poi_ori_train,random_poison_idx,best_noise.cpu(),transform_after_train)
print('Traing dataset size is:',len(poison_train_target)," Poison numbers is:",len(random_poison_idx))
clean_train_loader = DataLoader(poison_train_target, batch_size=test_batch_size, shuffle=True, num_workers=2)

# %%
#Attack success rate testing
test_non_target = list(np.where(np.array(test_label)!=lab)[0])
test_non_target_change_image_label = poison_image_label(poi_ori_test,test_non_target,best_noise.cpu()*multi_test,lab,None)
asr_loaders = torch.utils.data.DataLoader(test_non_target_change_image_label, batch_size=test_batch_size, shuffle=True, num_workers=2)
print('Poison test dataset size is:',len(test_non_target_change_image_label))

# %%
#Clean acc test dataset
clean_test_loader = torch.utils.data.DataLoader(ori_test, batch_size=test_batch_size, shuffle=False, num_workers=2)

# %%
#Target clean test dataset
test_target = list(np.where(np.array(test_label)==lab)[0])
target_test_set = Subset(ori_test,test_target)
target_test_loader = torch.utils.data.DataLoader(target_test_set, batch_size=test_batch_size, shuffle=True, num_workers=2)

# %%
from util import AverageMeter
train_ACC = []
test_ACC = []
clean_ACC = []
target_ACC = []

# %%
for epoch in tqdm(range(training_epochs)):
    # Train
    model.train()
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    pbar = tqdm(clean_train_loader, total=len(clean_train_loader))
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
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
    train_ACC.append(acc_meter.avg)
    print('Train_loss:',loss)
    scheduler.step()
    
    # Testing attack effect
    model.eval()
    correct, total = 0, 0
    for i, (images, labels) in enumerate(asr_loaders):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(images)
            out_loss = criterion(logits,labels)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    test_ACC.append(acc)
    print('\nAttack success rate %.2f' % (acc*100))
    print('Test_loss:',out_loss)
    
    correct_clean, total_clean = 0, 0
    for i, (images, labels) in enumerate(clean_test_loader):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(images)
            out_loss = criterion(logits,labels)
            _, predicted = torch.max(logits.data, 1)
            total_clean += labels.size(0)
            correct_clean += (predicted == labels).sum().item()
    acc_clean = correct_clean / total_clean
    clean_ACC.append(acc_clean)
    print('\nTest clean Accuracy %.2f' % (acc_clean*100))
    print('Test_loss:',out_loss)
    
    correct_tar, total_tar = 0, 0
    for i, (images, labels) in enumerate(target_test_loader):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(images)
            out_loss = criterion(logits,labels)
            _, predicted = torch.max(logits.data, 1)
            total_tar += labels.size(0)
            correct_tar += (predicted == labels).sum().item()
    acc_tar = correct_tar / total_tar
    target_ACC.append(acc_tar)
    print('\nTarget test clean Accuracy %.2f' % (acc_tar*100))
    print('Test_loss:',out_loss)

# %%
#ours -- higher_configureations
from matplotlib import pyplot as plt
half = np.arange(0,training_epochs)
plt.figure(figsize=(12.5,8))
plt.plot(half, np.asarray(train_ACC)[half], label='Training ACC', linestyle="-.", marker="o", linewidth=3.0, markersize = 8)
plt.plot(half, np.asarray(test_ACC)[half], label='Attack success rate', linestyle="-.", marker="o", linewidth=3.0, markersize = 8)
plt.plot(half, np.asarray(clean_ACC)[half], label='Clean test ACC', linestyle="-.", marker="o", linewidth=3.0, markersize = 8)
plt.plot(half, np.asarray(target_ACC)[half], label='Target class clean test ACC', linestyle="-", marker="o", linewidth=3.0, markersize = 8)
# plt.plot(half, np.asarray(test_unl_ACC)[half], label='protected test ACC', linestyle="-.", marker="o", linewidth=3.0, markersize = 8)
plt.ylabel('ACC', fontsize=24)
plt.xticks(fontsize=20)
plt.xlabel('Epoches', fontsize=24)
plt.yticks(np.arange(0,1.1, 0.1),fontsize=20)
plt.legend(fontsize=20,bbox_to_anchor=(1.016, 1.2),ncol=2)
plt.grid(color="gray", linestyle="-")
# plt.show()
plt.savefig("test_600_target_20230428.png")

dis_idx = clean_ACC.index(max(clean_ACC))
print(train_ACC[dis_idx])
print('attack',test_ACC[dis_idx])
print(clean_ACC.index(max(clean_ACC)))
print('all class clean', clean_ACC[dis_idx])
print('target clean',target_ACC[dis_idx])


