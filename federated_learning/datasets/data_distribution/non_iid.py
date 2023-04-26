import torch
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
# from torch.utils.data import DataLoader
import numpy as np


def generate_non_iid_data(train_dataset, test_dataset, args):
    """
        train_dataloader
        test_dataloader
    """
    # partition_method = args.partition_method
    # list object to numpy
    y_train = np.array(train_dataset.targets)
    # y_train = train_dataset.targets.umpy()

    number_of_classs = len(np.unique(y_train))
    n_train = len(train_dataset)
    n_nets = args.num_workers # #total clients
        
    # if partition_method == "hetero-dir":
    partition_alpha = args.partition_alpha
    min_size = 0
    min_required_size = 40 # #samples/client
    K = number_of_classs # number of classes
    dataset = args.dataset # Cifar10
    net_dataidx_map = {}

    while (min_size < min_required_size) or (dataset == 'mnist' and min_size < 100):
        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(partition_alpha, n_nets))
            ## Balance
            proportions = np.array([p*(len(idx_j) < n_train/n_nets) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
        # net_dataidx_map[j] = idx_batch[j][:500]
    # elif partition_method == "homo":
    # 	print("Go to this {} method".format(partition_method))
    # 	idxs = np.random.permutation(n_train)
    # 	batch_idxs = np.array_split(idxs, n_nets)
    # 	net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}
    # count how many samples in each client
    total_sample = 0
    for j in range(n_nets):
        print("Client %d: %d samples" % (j, len(net_dataidx_map[j])))
        cnt_class = {}
        for i in net_dataidx_map[j]:
            label = y_train[i]
            if label not in cnt_class:
                cnt_class[label] = 0
            cnt_class[label] += 1
        total_sample += len(net_dataidx_map[j])
        print("Client %d: %s" % (j, str(cnt_class)))
        print("--------"*10)
    print("Total training: %d samples" % total_sample)
    print("Total testing: %d samples" % len(test_dataset))
    # import IPython; IPython.embed()
    train_loaders = [
        torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            # sampler=SubsetRandomSampler(indices), # For random sampling
            sampler=SequentialSampler(indices),
        )
        for _, indices in net_dataidx_map.items()
    ]
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    # print(len(train_loaders[0]), len(test_loader))
    # exit(0)
    return train_loaders, test_loader, net_dataidx_map

def distribute_non_iid(train_loaders):
    distributed_dataset = [[] for _ in range(len(train_loaders))]
    for idx, train_loader in enumerate(train_loaders):
        for data, label in train_loader:
            distributed_dataset[idx].append((data, label))
    return distributed_dataset

