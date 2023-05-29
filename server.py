import os
import time
from tqdm import tqdm
from loguru import logger
from federated_learning.arguments import Arguments
from federated_learning.utils import generate_data_loaders_from_distributed_dataset
from federated_learning.datasets.data_distribution import distribute_batches_equally
from federated_learning.datasets.data_distribution import generate_non_iid_data
from federated_learning.datasets.data_distribution import distribute_non_iid

from federated_learning.utils.plot import plot_data_dis_to_file
from federated_learning.utils import average_nn_parameters
from federated_learning.utils import convert_distributed_data_into_numpy
from federated_learning.utils import poison_data
from federated_learning.utils import identify_random_elements
from federated_learning.utils import save_results
from federated_learning.utils import load_train_data_loader
from federated_learning.utils import load_test_data_loader
from federated_learning.utils import generate_experiment_ids
from federated_learning.utils import convert_results_to_csv
from federated_learning.utils import convert_results_to_csv_asr_cleanacc_taracc
from client import Client
from generate_train_test import get_dataset
from federated_learning.utils import get_labels
from federated_learning.utils import concate_dataset
from federated_learning.utils import apply_noise_patch
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
# import pysnooper
import argparse
import wandb
import pandas as pd
from federated_learning.nets import ResNet18
from copy import deepcopy


def train_poisoned_worker(epoch, args, client_idx, clients, poisoned_workers, target_label, dataset_pood="./data/"):
    args.get_logger().info("Training epoch #{} on poisoned client #{}", str(epoch), str(client_idx))
    best_noise = narcissus_gen(args, epoch, dataset_pood, client_idx, clients, target_label, poisoned_workers)

    return best_noise


def narcissus_gen(args, comm_round, dataset_path, client_idx, clients, target_label, poisoned_workers): # POOD + client dataset
    idx = poisoned_workers.index(client_idx)
    target_class = target_label[idx]

    n_channels = args.args_dict.narcissus_gen.n_channels
    #Noise size, default is full image size
    # noise_size = 32
    noise_size = args.args_dict.narcissus_gen.noise_size

    # checkpoint_path = "./checkpoint" # store surrogate models and best noise
    checkpoint_path = args.args_dict.narcissus_gen.checkpoint_path

    # If the checkpoint path doesn't exist, create one
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    
    device = args.device


    best_noise_prefix = args.args_dict.narcissus_gen.saving_best_noise_prefix
    exp_id = args.args_dict.fl_training.experiment_id
    gen_trigger_interval = args.args_dict.fl_training.gen_trigger_interval

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
        best_noise = torch.zeros((1, n_channels, noise_size, noise_size), device=device)
        noise_npy = np.load(best_noise_save_path)
        best_noise = torch.from_numpy(noise_npy).cuda()
        print(best_noise_save_path + " loaded")
        return best_noise
    
    client_train_loader = clients[client_idx].train_data_loader

    
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


    generating_model = args.net().cuda() # default: ResNet18_201

    #Surrogate model training epochs
    # surrogate_epochs = 200
    surrogate_epochs = args.args_dict.narcissus_gen.surrogate_epochs
    # surrogate_epochs = 300

    #Learning rate for poison-warm-up
    # generating_lr_warmup = 0.1
    generating_lr_warmup = args.args_dict.narcissus_gen.generating_lr_warmup
    # warmup_round = 5
    warmup_round = args.args_dict.narcissus_gen.warmup_round

    #Learning rate for trigger generating
    # generating_lr_tri = 0.01
    generating_lr_tri = args.args_dict.narcissus_gen.generating_lr_tri
    # gen_round = 1000
    gen_round = args.args_dict.narcissus_gen.gen_round

    #Training batch size
    # train_batch_size = 10
    train_batch_size = args.args_dict.narcissus_gen.train_batch_size

    #The model for adding the noise
    # patch_mode = "add"
    patch_mode = args.args_dict.narcissus_gen.patch_mode

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

    surrogate_loader = torch.utils.data.DataLoader(concate_train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=16)

    poi_warm_up_loader = torch.utils.data.DataLoader(train_target, batch_size=train_batch_size, shuffle=True, num_workers=16)

    trigger_gen_loaders = torch.utils.data.DataLoader(train_target, batch_size=train_batch_size, shuffle=True, num_workers=16)


    # surrogate_model = surrogate_model
    criterion = torch.nn.CrossEntropyLoss()
    # outer_opt = torch.optim.RAdam(params=base_model.parameters(), lr=generating_lr_outer)
    if args.args_dict.narcissus_gen.optimizer == "adamw":
        surrogate_opt = torch.optim.AdamW(params=surrogate_model.parameters(), lr=0.001, weight_decay=0.01)
    elif args.args_dict.narcissus_gen.optimizer == "sgd":
        surrogate_opt = torch.optim.SGD(params=surrogate_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        surrogate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(surrogate_opt, T_max=surrogate_epochs)

    # if not os.path.isfile(surrogate_pretrained_path):
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
            wandb.log({"surrogate_epoch": epoch, "surrogate_loss": ave_loss})


    
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

    if args.args_dict.narcissus_gen.optimizer == "radam":
        poi_warm_up_opt = torch.optim.RAdam(params=poi_warm_up_model.parameters(), lr=generating_lr_warmup)
    elif args.args_dict.narcissus_gen.optimizer == "adamw":
        poi_warm_up_opt = torch.optim.AdamW(params=poi_warm_up_model.parameters(), lr=generating_lr_warmup)

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
            loss.backward(retain_graph=True)
            loss_list.append(float(loss.data))
            poi_warm_up_opt.step()
        ave_loss = np.average(np.array(loss_list))
        print('Epoch:%d, Loss: %e' % (epoch, ave_loss))
        wandb.log({"poi_warm_up_epoch": epoch, "poi_warm_up_loss": ave_loss})

    #Trigger generating stage
    for param in poi_warm_up_model.parameters():
        param.requires_grad = False

    batch_pert = torch.autograd.Variable(noise.cuda(), requires_grad=True)

    if args.args_dict.narcissus_gen.optimizer == "radam":
        batch_opt = torch.optim.RAdam(params=[batch_pert], lr=generating_lr_tri)
    elif args.args_dict.narcissus_gen.optimizer == "adamw":
        batch_opt = torch.optim.AdamW(params=[batch_pert], lr=generating_lr_tri)
        
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
        wandb.log({"gen_round": round, "gradient": ave_grad, "trigger_gen_loss": ave_loss})
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
    

def train_subset_of_clients(epoch, args, clients, poisoned_workers, n_target_samples, global_model):
    """
    Train a subset of clients per round.

    :param epoch: epoch
    :type epoch: int
    :param args: arguments
    :type args: Arguments
    :param clients: clients
    :type clients: list(Client)
    :param poisoned_workers: indices of poisoned workers
    :type poisoned_workers: list(int)
    """
    # with pysnooper.snoop():
    kwargs = args.get_round_worker_selection_strategy_kwargs()
    kwargs["current_epoch_number"] = epoch

    # print(args.get_round_worker_selection_strategy())

    # random_workers = args.get_round_worker_selection_strategy().select_round_workers(
    #     list(range(args.get_num_workers())),
    #     poisoned_workers,
    #     kwargs)
    random_workers = args.get_round_worker_selection_strategy().select_round_workers_and_malicious_client(
        list(range(args.get_num_workers())),
        poisoned_workers,
        kwargs)
    # random_workers = args.get_round_worker_selection_strategy().select_2_poisoned_clients(
    #     list(range(args.get_num_workers())),
    #     poisoned_workers,
    #     kwargs)

    
    target_label = args.args_dict.fl_training.target_label # [2, 9] = [bird, truck]
    
    # noise_size = 32
    noise_size = args.args_dict.narcissus_gen.noise_size
    n_channels = args.args_dict.narcissus_gen.n_channels
    best_noise_dict = {} # poisoned_client_idx: best_noise

    for client_idx in random_workers:
        args.get_logger().info("Training epoch #{} on client #{}", str(epoch), str(clients[client_idx].get_client_index()))
        if client_idx in poisoned_workers:
            best_noise = train_poisoned_worker(epoch, args, client_idx, clients, poisoned_workers, target_label=target_label, dataset_pood="./data/") # NARCISSUS, target label: bird (CIFAR-10)
            best_noise_dict[client_idx] = best_noise
        else:
            best_noise_dict[client_idx] = None # if the client is not poisoned, then best_noise is None
        clients[client_idx].train(epoch, best_noise=best_noise_dict[client_idx], n_target_samples=n_target_samples, target_label=target_label) # trains clients, including the poisoned one (expected high clean ACC)
        # wandb.log({"comm_round": epoch, "client_idx": client_idx, "loss": loss})

    args.get_logger().info("Averaging client parameters")
    parameters = [clients[client_idx].get_nn_parameters() for client_idx in random_workers]
    # parameters = [] # scaling up local models' params 20 times
    # for client_idx in random_workers:
    #     for k, v in clients[client_idx].get_nn_parameters().items():
    #         clients[client_idx].get_nn_parameters()[k] = v * 20
    #     parameters.append(clients[client_idx].get_nn_parameters())
    
    new_nn_params = average_nn_parameters(parameters)
    global_model.load_state_dict(deepcopy(new_nn_params), strict=True)

    # clients[0].update_nn_parameters(new_nn_params)
    # clients[1].update_nn_parameters(new_nn_params)

    for client in clients:
        args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
        client.update_nn_parameters(new_nn_params)


    # return clients[3].test(best_noise=best_noise, target_label=2), random_workers
    # acc, acc_clean, acc_tar = clients[poisoned_workers[0]].test(best_noise=best_noise, target_label=target_label)
    results = []
    for poisoned_worker in poisoned_workers:
        acc, acc_clean, acc_tar = clients[poisoned_worker].test(best_noise=best_noise_dict[poisoned_worker], n_target_samples=n_target_samples, target_label=target_label)
        results.append((acc, acc_clean, acc_tar))
    
    # acc0, acc_clean0, acc_tar0 = clients[poisoned_workers[0]].test(best_noise=best_noise_dict[poisoned_workers[0]], target_label=target_label)
    # acc1, acc_clean1, acc_tar1 = clients[poisoned_workers[1]].test(best_noise=best_noise_dict[poisoned_workers[1]], target_label=target_label)

    # acc0, acc_clean0, acc_tar0 = clients[0].test(best_noise=best_noise_dict[poisoned_workers[0]], target_label=target_label)
    # acc0, acc_clean0, acc_tar0 = clients[1].test(best_noise=best_noise_dict[poisoned_workers[0]], target_label=target_label)
    # acc1, acc_clean1, acc_tar1 = clients[poisoned_workers[1]].test(best_noise=best_noise_dict[poisoned_workers[1]], target_label=target_label)
    # wandb.log({"comm_round": epoch, "acc": acc, "acc_clean": acc_clean, "acc_tar": acc_tar})
    # return (acc, acc_clean, acc_tar), random_workers # Compute metrics on the poisoned client
    # return (acc0, acc_clean0, acc_tar0), (acc1, acc_clean1, acc_tar1), random_workers # Compute metrics on the poisoned client
    return results, random_workers # Compute metrics on the poisoned client

def create_clients(args, train_data_loaders, test_data_loader, poisoned_workers, global_model):
    """
    Create a set of clients.
    """
    clients = []
    for idx in range(args.get_num_workers()):
        clients.append(Client(args, idx, train_data_loaders[idx], test_data_loader, poisoned_workers, global_model))

    return clients

def run_machine_learning(clients, args, poisoned_workers, n_target_samples, global_model):
    """
    Complete machine learning over a series of clients.
    """
    wandb_name = f"{args.args_dict.fl_training.wandb_name}__num_workers_{args.num_workers}__num_selected_workers_{args.num_workers}__num_poisoned_workers_{args.get_num_poisoned_workers()}__poison_amount_ratio_{args.args_dict.narcissus_gen.poison_amount_ratio}__local_epochs_{args.args_dict.fl_training.local_epochs}__target_label_{args.args_dict.fl_training.target_label}__poisoned_workers_{args.args_dict.fl_training.poisoned_workers}__n_target_samples_{args.args_dict.fl_training.n_target_samples}__multi_test_{args.args_dict.narcissus_gen.multi_test}__patch_mode_{args.args_dict.narcissus_gen.patch_mode}__gen_round_{args.args_dict.narcissus_gen.gen_round}__gen_trigger_interval_{args.args_dict.fl_training.gen_trigger_interval}__narcissus_optimizer_{args.args_dict.narcissus_gen.optimizer}__exp_{args.args_dict.fl_training.experiment_id}"
    wandb.init(name=wandb_name, project=args.args_dict.fl_training.project_name, entity="nguyenhongsonk62hust")

    # epoch_test_set_results = []
    # epoch_test_set_results0 = []
    # epoch_test_set_results1 = []
    epoch_test_set_results = []
    worker_selection = []

    # global_model = ResNet18().cuda()

    for epoch in range(1, args.get_num_epochs() + 1): # communication rounds
        # Reinitialize the local model
        for client in clients:
            client.reinitialize_after_each_round(global_model)
            
        # results, workers_selected = train_subset_of_clients(epoch, args, clients, poisoned_workers)
        # results0, results1, workers_selected = train_subset_of_clients(epoch, args, clients, poisoned_workers)
        # acc0, acc_clean0, acc_tar0 = results0
        # acc1, acc_clean1, acc_tar1 = results1
        results, workers_selected = train_subset_of_clients(epoch, args, clients, poisoned_workers, n_target_samples, global_model)
        # wandb.log({"comm_round": epoch, "asr": acc, "acc_clean": acc_clean, "acc_tar": acc_tar})
        # wandb.log({"comm_round__client_0": epoch, "asr__client_0": acc0, "acc_clean__client_0": acc_clean0, "acc_tar__client_0": acc_tar0})
        # wandb.log({"comm_round__client_1": epoch, "asr__client_1": acc1, "acc_clean__client_1": acc_clean1, "acc_tar__client_1": acc_tar1})
        for i in range(len(poisoned_workers)):
            acc, acc_clean, acc_tar = results[i]
            wandb.log({"comm_round__client_" + str(poisoned_workers[i]): epoch, "asr__client_" + str(poisoned_workers[i]): acc, "acc_clean__client_" + str(poisoned_workers[i]): acc_clean, "acc_tar__client_" + str(poisoned_workers[i]): acc_tar})
            

        # epoch_test_set_results.append(results)
        # epoch_test_set_results0.append(results0)
        # epoch_test_set_results1.append(results1)
        for result in results:
            epoch_test_set_results.append(result)
        worker_selection.append(workers_selected)

    # converted_epoch_test_set_results = []
    # for result in epoch_test_set_results:
    #     converted_epoch_test_set_results.append(convert_results_to_csv_asr_cleanacc_taracc(result))

    # return convert_results_to_csv(epoch_test_set_results), worker_selection
    # return convert_results_to_csv_asr_cleanacc_taracc(epoch_test_set_results), worker_selection
    # return convert_results_to_csv_asr_cleanacc_taracc(epoch_test_set_results0), convert_results_to_csv_asr_cleanacc_taracc(epoch_test_set_results1), worker_selection
    return epoch_test_set_results, worker_selection

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
        for j in range(args.num_workers):
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
        n_target_samples.append(max_index[1][1])
    return poisoned_workers, n_target_samples

def run_exp(KWARGS, client_selection_strategy, idx):
    log_files, results_files, models_folders, worker_selections_files = generate_experiment_ids(idx, 1)

    # Initialize logger
    handler = logger.add(log_files[0], enqueue=True)

    parser = argparse.ArgumentParser(description="A Clean-Label Attack in FL")
    parser.add_argument("--config", type=str, help="Configuration file", default="federated_learning/config/test.json")

    config = parser.parse_args().config
    absolute_config_path = os.path.join(os.getcwd(), config)

    args = Arguments(logger, config_filepath=absolute_config_path)
    args.set_model_save_path(models_folders[0])
    # args.set_num_poisoned_workers(num_poisoned_workers)
    args.set_round_worker_selection_strategy_kwargs(KWARGS)
    args.set_client_selection_strategy(client_selection_strategy)
    args.log()

    kwargs = {"num_workers": 0, "pin_memory": True} if args.cuda else {}
    # train_dataset, test_dataset, train_data_loader, test_data_loader = get_dataset(args, kwargs)
    train_dataset, test_dataset = get_dataset(args, kwargs)

    # train_data_loader = load_train_data_loader(logger, args)
    # test_data_loader = load_test_data_loader(logger, args)


    # Distribute batches equal volume IID (IID distribution)
    # distributed_train_dataset = distribute_batches_equally(train_data_loader, args.get_num_workers())
    kwargs = {"num_workers": 4, "pin_memory": True} if args.cuda else {}
    # train_loaders, test_loader, net_dataidx_map = generate_non_iid_data(train_dataset, test_dataset, args)
    train_loaders, test_data_loader, net_dataidx_map = generate_non_iid_data(train_dataset, test_dataset, args, kwargs)

    # import IPython
    # plot data distribution by matplotlib
    # count sample for each class in each worker
    labels_clients = {}

    for i in range(args.get_num_workers()):
        cnt_class = {}
        for j in net_dataidx_map[i]:
            label = train_dataset.targets[j]
            if label not in cnt_class:
                cnt_class[label] = 0
            cnt_class[label] += 1
        print("Client %d: %d samples" % (i, len(net_dataidx_map[i])))
        # print(cnt_class)
        labels_clients[f"Client {i}"] = cnt_class

    data_file_name = f"non_iid_data_distribution__dataset_{args.args_dict.fl_training.dataset.lower()}__partition_alpha_{args.args_dict.fl_training.partition_alpha}__num_workers_{args.args_dict.fl_training.num_workers}__exp_{args.args_dict.fl_training.experiment_id}.png"
    plot_data_dis_to_file(labels_clients, num_class=10, data_file_name=os.path.join("./plots", data_file_name))


    # IPython.embed()

    # exit(0)

    # plot 
    # distributed_train_dataset = distribute_non_iid(train_loaders)
    # distributed_train_dataset = convert_distributed_data_into_numpy(distributed_train_dataset) # review, why do we need to convert?
    
    # poisoned_workers = identify_random_elements(args.get_num_workers(), args.get_num_poisoned_workers())
    # distributed_train_dataset = poison_data(logger, distributed_train_dataset, args.get_num_workers(), poisoned_workers, replacement_method)

    # train_data_loaders = generate_data_loaders_from_distributed_dataset(distributed_train_dataset, args.get_batch_size()) # review

    # poisoned_workers = [0]

    # poisoned_workers = select_poisoned_workers(args, train_dataset, net_dataidx_map)
    poisoned_workers, n_target_samples = select_poisoned_workers(args, train_dataset, net_dataidx_map)
    # poisoned_workers = [0, 1]
    # poison_class = [0, 4] # args.args_dict.fl_training.target_label
    # poisoned_workers = [0, 1]
    poisoned_workers = args.args_dict.fl_training.poisoned_workers
    # n_target_samples = [2538, 1777]
    n_target_samples = args.args_dict.fl_training.n_target_samples
    print("Poisoned workers: ", poisoned_workers)
    print("Number of target samples: ", n_target_samples)
    # exit(0)
    # tmp.sort(key=lambda x: x[1], reverse=True)
    # np.argmax(tmp, key=)
    # index = tmp[0]
    #     print("Client %d: %s" % (j, str(cnt_class)))
    #     print("--------"*10)
    # print("Total training: %d samples" % total_sample)
    # print("Total testing: %d samples" % len(test_dataset))
    # for j in range(args.num_workers):
    #     for k, v in cnt_class.items():
    #         list(cnt)

    global_model = ResNet18().cuda()

    # clients = create_clients(args, train_data_loaders, test_data_loader)
    # clients = create_clients(args, train_data_loaders, test_data_loader, poisoned_workers)
    clients = create_clients(args, train_loaders, test_data_loader, poisoned_workers, global_model)

    # results, worker_selection = run_machine_learning(clients, args, poisoned_workers)
    # results0, results1, worker_selection = run_machine_learning(clients, args, poisoned_workers)
    results, worker_selection = run_machine_learning(clients, args, poisoned_workers, n_target_samples, global_model)
    # save_results(results, results_files[0])
    # save_results(worker_selection, worker_selections_files[0])

    # create a dataframe from worker_selection = [[1,2,3,4], [2,3,4,5]]
    # worker_selection_df = pd.DataFrame(worker_selection)

    # table =  wandb.Table(dataframe=worker_selection_df)
    # wandb.log({"workers_selected": table})

    logger.remove(handler)
