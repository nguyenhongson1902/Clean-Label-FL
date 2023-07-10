import os
import numpy as np
import argparse
from loguru import logger
from tqdm import tqdm
import pickle
from copy import deepcopy

import torch
from torch.utils.data import Subset
import torchvision.transforms as transforms
import torchvision

from federated_learning.utils import get_labels
from federated_learning.utils import concate_dataset
from federated_learning.utils import FlippedDataset
from federated_learning.utils import apply_noise_patch
from federated_learning.arguments import Arguments


SEED=1
torch.manual_seed(SEED)
np.random.seed(SEED)


def train_poisoned_worker(args, client_idx, target_label, train_data_loader, pood_path="./data/"):
    """
    Train the model on the poisoned dataset to get an optimized trigger
    :param client_idx: index of the poisoned client
    :type client_idx: int
    :param target_label: target label of the poisoned client
    :type target_label: list
    :param pood_path: path to the POOD dataset (e.g: TinyImageNet)
    :type pood_path: str
    :return: best_noise: an optimized trigger
    :type best_noise: numpy.ndarray
    """

    args.get_logger().info("Training on poisoned client #{}", str(client_idx))
    best_noise = narcissus_gen(args, pood_path, client_idx, target_label, train_data_loader)
    return best_noise

def narcissus_gen(args, dataset_path, client_idx, target_label, train_data_loader): # POOD + client dataset
    """
    The function `narcissus_gen` generates a trigger (also known as a backdoor or poison) for a specific
    client in a federated learning setting.
    
    :param args: The `args` parameter is an object that contains various arguments and configurations
    for the function. It is used to pass information and settings to the function
    :param dataset_path: The `dataset_path` parameter is the path to the directory where the dataset is
    stored. It should contain the subdirectory "tiny-imagenet-200/train/" which contains the POOD images
    :param client_idx: The index of the client for which the trigger is being generated
    :param target_label: The `target_label` parameter is a list that contains the target label for each
    poisoned worker. Each element in the list represents the target label for a specific poisoned
    worker.
    :param train_data_loader: The `train_data_loader` parameter is the data loader for the client's
    training dataset. It is used to iterate over the training data during the surrogate model training,
    poi_warm_up training, and trigger generation stages
    :return: the generated noise tensor.
    """

    device = args.device
    poisoned_workers = args.args_dict.fl_training.poisoned_workers # [0]
    idx = poisoned_workers.index(client_idx) # index of the poisoned client
    target_class = target_label[idx] # e.g: target label 2 (bird)

    n_channels = args.args_dict.narcissus_gen.n_channels # Default: 3
    noise_size = args.args_dict.narcissus_gen.noise_size # Default: 32

    checkpoint_path = args.args_dict.narcissus_gen.checkpoint_path # store surrogate models and best noise
    exp_id = args.args_dict.fl_training.experiment_id

    # If the checkpoint path doesn't exist, create one
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    exp_id = args.args_dict.fl_training.experiment_id
    best_noise_prefix = args.args_dict.narcissus_gen.saving_best_noise_prefix

    best_noise_save_path = os.path.join(checkpoint_path, best_noise_prefix + "__client_" + str(client_idx) + "__target_label_" + str(target_class) + "__exp_" + str(exp_id) + ".npy")
    
    client_train_loader = train_data_loader
    
    # Radius of the L-inf ball
    l_inf_r = args.args_dict.narcissus_gen.l_inf_r / 255 # Default: l_inf_r = 16/255

    # Model for generating surrogate model and trigger
    surrogate_model = args.net().to(device) # default: ResNet18_201

    # surrogate_pretrained_path = os.path.join(checkpoint_path, 'surrogate_pretrain_client_' + str(client_idx) + '_comm_round_' + str(comm_round) + '.pth')

    # generating_model = args.net().to(device) # default: ResNet18_201

    # Surrogate model training epochs
    surrogate_epochs = args.args_dict.narcissus_gen.surrogate_epochs # Default: surrogate_epochs = 200

    # Learning rate for poison-warm-up
    generating_lr_warmup = args.args_dict.narcissus_gen.generating_lr_warmup # Default: # generating_lr_warmup = 0.1
    
    warmup_round = args.args_dict.narcissus_gen.warmup_round # Default: warmup_round = 5

    # Learning rate for trigger generating
    generating_lr_tri = args.args_dict.narcissus_gen.generating_lr_tri # Default: generating_lr_tri = 0.01
    
    gen_round = args.args_dict.narcissus_gen.gen_round # Default: gen_round = 1000

    # Training batch size
    train_batch_size = args.args_dict.narcissus_gen.train_batch_size # Default: train_batch_size = 10

    # The model for adding the noise
    patch_mode = args.args_dict.narcissus_gen.patch_mode # Default: patch_mode = "add"

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

    #######################Test flipping TinyImageNet + target examples (CIFAR-10)###############################
    # X, Y = [], []
    # for x, y in tqdm(concate_train_dataset, total=len(concate_train_dataset), desc="Flipping labels"):
    #     X.append(x)
    #     # Flip the label
    #     if y != 200: # class 200 is the label of CIFAR-10 target examples
    #         Y.append(200)
    #     else:
    #         Y.append(y)

    # flipped_dataset = FlippedDataset(X, Y)
    # concate_train_dataset = flipped_dataset # A trick to help not to change the variable below
    #############################################################################################################

    surrogate_loader = torch.utils.data.DataLoader(concate_train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)

    poi_warm_up_loader = torch.utils.data.DataLoader(train_target, batch_size=train_batch_size, shuffle=True, num_workers=0)

    # trigger_gen_loaders = torch.utils.data.DataLoader(train_target, batch_size=train_batch_size, shuffle=True, num_workers=0)

    # Create a duplicated target examples dataset
    train_target_list_duplicated = list(np.where(np.array(train_label)==target_class)[0]) * 10 # duplicate 10 times
    train_target_duplicated = Subset(ori_train, train_target_list_duplicated)

    trigger_gen_loaders = torch.utils.data.DataLoader(train_target_duplicated, batch_size=train_batch_size, shuffle=True, num_workers=0)


    criterion = torch.nn.CrossEntropyLoss()
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
    saving_surrogate_model_prefix = args.args_dict.narcissus_gen.saving_surrogate_model_prefix
    surrogate_pretrained_path = os.path.join(checkpoint_path, saving_surrogate_model_prefix + "__client_" + str(client_idx) + "__target_label_" + str(target_class) + "__exp_" + str(exp_id) + ".pth")

    if os.path.exists(surrogate_pretrained_path):
        print("Loading the surrogate model...")
        surrogate_model.load_state_dict(torch.load(surrogate_pretrained_path))
        print("Done loading!!")
    else:
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
            if args.args_dict.narcissus_gen.optimizer == "sgd":
                surrogate_scheduler.step()
            ave_loss = np.average(np.array(loss_list))
            print('Epoch:%d, Loss: %.03f' % (epoch, ave_loss))
        # Save the surrogate model
        print("Saving the surrogate model...")
        torch.save(surrogate_model.state_dict(), surrogate_pretrained_path)
        print("Done saving!!")


    # After getting the pre-trained surrogate model, we start to train the poi_warm_up model
    # Prepare models and optimizers for poi_warm_up training
    poi_warm_up_model = deepcopy(surrogate_model) # use poi_warm_up_model for generating the trigger later on
    # poi_warm_up_model.load_state_dict(surrogate_model.state_dict())

    if args.args_dict.narcissus_gen.optimizer == "radam" or args.args_dict.narcissus_gen.optimizer == "sgd":
        poi_warm_up_opt = torch.optim.RAdam(params=poi_warm_up_model.parameters(), lr=generating_lr_warmup)
    elif args.args_dict.narcissus_gen.optimizer == "adamw":
        poi_warm_up_opt = torch.optim.AdamW(params=poi_warm_up_model.parameters(), lr=generating_lr_warmup)

    # Poi_warm_up stage
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
        # if ave_grad == 0:
        #     break

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
    parser.add_argument("--client_idx", type=int, help="Client index", default=0) # poisoned client index

    config = parser.parse_args().config
    absolute_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config)

    args = Arguments(logger, config_filepath=absolute_config_path)
    args.log()

    # Load train loaders, test loader, train indices
    n_clients = args.args_dict.fl_training.num_clients
    train_loaders_path = f"./data_loaders/cifar10/iid/train_loaders_iid_n_clients_{n_clients}.pkl"
    test_data_loader_path = f"./data_loaders/cifar10/iid/test_data_loader_iid_n_clients_{n_clients}.pkl"
    train_indices_path = f"./data_loaders/cifar10/iid/train_indices_iid_n_clients_{n_clients}.pkl"
    with open(train_loaders_path, 'rb') as f:
        train_loaders = pickle.load(f)
    with open(test_data_loader_path, 'rb') as f:
        test_data_loader = pickle.load(f)
    with open(train_indices_path, 'rb') as f:
        train_indices = pickle.load(f)


    client_idx = parser.parse_args().client_idx
    target_label = args.args_dict.fl_training.target_label
    train_poisoned_worker(args, client_idx, target_label, train_loaders[client_idx], pood_path="./data/")