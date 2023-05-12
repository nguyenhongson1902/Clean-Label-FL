import os
import time
from tqdm import tqdm
from loguru import logger
from federated_learning.arguments import Arguments
from federated_learning.utils import generate_data_loaders_from_distributed_dataset
from federated_learning.datasets.data_distribution import distribute_batches_equally
from federated_learning.datasets.data_distribution import generate_non_iid_data
from federated_learning.datasets.data_distribution import distribute_non_iid

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
import pysnooper


def train_poisoned_worker(epoch, args, client_idx, clients, target_label):
    args.get_logger().info("Training epoch #{} on poisoned client #{}", str(epoch), str(client_idx))
    dataset_POOD = './data/'
    best_noise = narcissus_gen(args, epoch, dataset_POOD, client_idx, clients[client_idx].train_data_loader, target_label)

    return best_noise


def narcissus_gen(args, comm_round, dataset_path, client_idx, client_train_loader, target_label): # POOD + client dataset
    if torch.cuda.is_available() and args.get_cuda():
        device = "cuda"
    else:
        device = "cpu"

    checkpoint_path = "./checkpoint"

    #Noise size, default is full image size
    noise_size = 32

    #Radius of the L-inf ball
    l_inf_r = 16/255

    #Model for generating surrogate model and trigger
    surrogate_model = args.net().cuda() # default: ResNet18_201

    save_name = os.path.join(checkpoint_path, 'best_noise_client_' + str(client_idx) + '.npy')
    if os.path.isfile(save_name):
        best_noise = torch.zeros((1, 3, noise_size, noise_size), device=device)
        noise_npy = np.load(save_name)
        best_noise = torch.from_numpy(noise_npy).cuda()
        return best_noise
    

    # surrogate_pretrained_path = os.path.join(checkpoint_path, 'surrogate_pretrain_client_' + str(client_idx) + '_comm_round_' + str(comm_round) + '.pth')
    surrogate_pretrained_path = os.path.join(checkpoint_path, 'surrogate_pretrain_client_' + str(client_idx) + '.pth')
    if os.path.isfile(surrogate_pretrained_path):
        surrogate_model.load_state_dict(torch.load(surrogate_pretrained_path))
        print("Loaded the pre-trained surrogate model")


    generating_model = args.net().cuda() # default: ResNet18_201

    #Surrogate model training epochs
    surrogate_epochs = 200
    # surrogate_epochs = 300

    #Learning rate for poison-warm-up
    generating_lr_warmup = 0.1
    warmup_round = 5

    #Learning rate for trigger generating
    generating_lr_tri = 0.01
    gen_round = 1000

    #Training batch size
    train_batch_size = 10

    #The model for adding the noise
    patch_mode = "add"

    #The arguments use for surrogate model training stage
    transform_surrogate_train = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
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

    ori_train = client_train_loader.dataset

    # ori_train = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform_train)
    # ori_test = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=False, transform=transform_test)
    outter_trainset = torchvision.datasets.ImageFolder(root=dataset_path + '/tiny-imagenet-200/train/', transform=transform_surrogate_train)

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
    noise = torch.zeros((1, 3, noise_size, noise_size), device=device)

    #Inner train dataset
    train_target_list = list(np.where(np.array(train_label)==target_label)[0])
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
    surrogate_opt = torch.optim.SGD(params=surrogate_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    surrogate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(surrogate_opt, T_max=surrogate_epochs)

    # if not os.path.isfile(surrogate_pretrained_path):
    # #Training the surrogate model
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

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    #Save the surrogate model
    # save_path = os.path.join(checkpoint_path, 'surrogate_pretrain_comm_round_' + str(comm_round) + '.pth')
    save_path = os.path.join(checkpoint_path, 'surrogate_pretrain_client_' + str(client_idx) + '.pth')
    # save_path = './checkpoint/surrogate_pretrain_comm_round_' + str(comm_round) + '.pth'
    torch.save(surrogate_model.state_dict(), save_path)

    #Prepare models and optimizers for poi_warm_up training
    poi_warm_up_model = generating_model
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
    for _ in tqdm(range(gen_round)):
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

    noise = torch.clamp(batch_pert, -l_inf_r*2, l_inf_r*2)
    best_noise = noise.clone().detach().cpu()

    #Save the trigger
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    # save_name = os.path.join(checkpoint_path, 'best_noise_client_' + str(client_idx) + '_' + 'round_' + str(comm_round) + '.npy')
    save_name = os.path.join(checkpoint_path, 'best_noise_client_' + str(client_idx) + '.npy')
    # save_name = './checkpoint/best_noise_client_'+str(client_idx)+'_'+'round_'+str(comm_round)
    np.save(save_name, best_noise)

    # plt.imshow(np.transpose(noise[0].detach().cpu(),(1,2,0)))
    # plt.show()
    # print('Noise max val:',noise.max())

    return noise.clone().detach() # don't move the tensor to CPU
    


def train_subset_of_clients(epoch, args, clients, poisoned_workers):
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

    noise_size = 32
    best_noise = torch.zeros((1, 3, noise_size, noise_size), device=args.device)
    for client_idx in random_workers:
        args.get_logger().info("Training epoch #{} on client #{}", str(epoch), str(clients[client_idx].get_client_index()))
        if client_idx in poisoned_workers:
            best_noise = train_poisoned_worker(epoch, args, client_idx, clients, target_label=2) # NARCISSUS, target label: bird (CIFAR-10)
        clients[client_idx].train(epoch, best_noise, target_label=2) # trains clients, including the poisoned one (expected high clean ACC)

    args.get_logger().info("Averaging client parameters")
    parameters = [clients[client_idx].get_nn_parameters() for client_idx in random_workers]
    # parameters = [] # scaling up local models' params 20 times
    # for client_idx in random_workers:
    #     for k, v in clients[client_idx].get_nn_parameters().items():
    #         clients[client_idx].get_nn_parameters()[k] = v * 20
    #     parameters.append(clients[client_idx].get_nn_parameters())
    
    new_nn_params = average_nn_parameters(parameters)

    for client in clients:
        args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
        client.update_nn_parameters(new_nn_params)

    # return clients[3].test(best_noise=best_noise, target_label=2), random_workers
    return clients[poisoned_workers[0]].test(best_noise=best_noise, target_label=2), random_workers

def create_clients(args, train_data_loaders, test_data_loader, poisoned_workers):
    """
    Create a set of clients.
    """
    clients = []
    for idx in range(args.get_num_workers()):
        clients.append(Client(args, idx, train_data_loaders[idx], test_data_loader, poisoned_workers))

    return clients

def run_machine_learning(clients, args, poisoned_workers):
    """
    Complete machine learning over a series of clients.
    """
    epoch_test_set_results = []
    worker_selection = []
    for epoch in range(1, args.get_num_epochs() + 1): # communication rounds
        results, workers_selected = train_subset_of_clients(epoch, args, clients, poisoned_workers)

        epoch_test_set_results.append(results)
        worker_selection.append(workers_selected)

    # return convert_results_to_csv(epoch_test_set_results), worker_selection
    return convert_results_to_csv_asr_cleanacc_taracc(epoch_test_set_results), worker_selection

def run_exp(replacement_method, num_poisoned_workers, KWARGS, client_selection_strategy, idx):
    log_files, results_files, models_folders, worker_selections_files = generate_experiment_ids(idx, 1)

    # Initialize logger
    handler = logger.add(log_files[0], enqueue=True)

    args = Arguments(logger)
    args.set_model_save_path(models_folders[0])
    args.set_num_poisoned_workers(num_poisoned_workers)
    args.set_round_worker_selection_strategy_kwargs(KWARGS)
    args.set_client_selection_strategy(client_selection_strategy)
    args.log()

    kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
    train_dataset, test_dataset, train_data_loader, test_data_loader = get_dataset(args, kwargs)

    # train_data_loader = load_train_data_loader(logger, args)
    # test_data_loader = load_test_data_loader(logger, args)


    # Distribute batches equal volume IID (IID distribution)
    distributed_train_dataset = distribute_batches_equally(train_data_loader, args.get_num_workers())
    # train_loaders, test_loader, net_dataidx_map = generate_non_iid_data(train_dataset, test_dataset, args)
    # distributed_train_dataset = distribute_non_iid(train_loaders)
    distributed_train_dataset = convert_distributed_data_into_numpy(distributed_train_dataset) # review, why do we need to convert?
    
    # poisoned_workers = identify_random_elements(args.get_num_workers(), args.get_num_poisoned_workers())
    # distributed_train_dataset = poison_data(logger, distributed_train_dataset, args.get_num_workers(), poisoned_workers, replacement_method)

    train_data_loaders = generate_data_loaders_from_distributed_dataset(distributed_train_dataset, args.get_batch_size()) # review

    # y_train = np.array(train_dataset.targets)
    # total_sample = 0
    # tmp = []
    # for j in range(args.num_workers):
    #     print("Client %d: %d samples" % (j, len(net_dataidx_map[j])))
    #     cnt_class = {}
    #     for i in net_dataidx_map[j]:
    #         label = y_train[i]
    #         if label not in cnt_class:
    #             cnt_class[label] = 0
    #         cnt_class[label] += 1
    #     total_sample += len(net_dataidx_map[j])

    #     # lst = list(cnt_class.items())
    #     lst = []
    #     for t in cnt_class.items():
    #         if t[0]==2:
    #             lst.append(t)
    #             break
    #     if not lst:
    #         lst.append((2, 0)) # did not find any examples with label 2

    #     tmp.extend(lst)

    # max_index = max(enumerate(tmp), key=lambda x: x[1][1])
    # poisoned_workers = [max_index[0]]


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

    poisoned_workers = [0]

    # clients = create_clients(args, train_data_loaders, test_data_loader)
    clients = create_clients(args, train_data_loaders, test_data_loader, poisoned_workers)
    # clients = create_clients(args, train_loaders, test_data_loader, poisoned_workers)

    results, worker_selection = run_machine_learning(clients, args, poisoned_workers)
    save_results(results, results_files[0])
    save_results(worker_selection, worker_selections_files[0])

    logger.remove(handler)
