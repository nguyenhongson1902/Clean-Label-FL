import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn

from federated_learning.utils import get_labels
from federated_learning.utils import poison_image_label
import numpy as np
from torch.utils.data import Subset
import glob


SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)


def server_test_fn(
    args,
    global_model, 
    client_idx,
    device=torch.device("cpu"), 
):
    """
    The function `server_test_fn` performs server testing by evaluating the attack success rate (ASR),
    clean accuracy, and target accuracy of a global model on a test dataset.
    
    :param args: The `args` parameter is an object that contains various arguments and configurations
    for the server testing function. It is used to access specific values needed for the testing process
    :param global_model: The `global_model` parameter is the model that will be used for server testing.
    It should be an instance of a PyTorch model that has been trained on the server using federated
    learning
    :param client_idx: The `client_idx` parameter represents the index of the client for which the
    server is performing the testing. It is used to determine the target label for poisoning and to
    identify the corresponding noise file
    :param device: The "device" parameter is used to specify the device on which the model will be
    evaluated. It can be set to either "cpu" or a specific GPU device (e.g., "cuda:0") if available
    :return: a dictionary containing the attack success rate (ASR), clean accuracy (clean_acc), and
    target test accuracy (tar_acc).
    """
    
    print("\nStart Server Testing ...\n" + " = "*16)

    # Load the saved best noise
    checkpoint_path = args.args_dict.narcissus_gen.checkpoint_path
    exp_id = args.args_dict.fl_training.experiment_id
    # Specify the pattern to match
    pattern = '*exp_{}.npy'.format(exp_id)
    # Get a list of files matching the pattern
    matching_files = glob.glob(os.path.join(checkpoint_path, pattern))
    # Check if any files match the pattern
    if len(matching_files) > 0:
        # Load the first matching file
        file_path = matching_files[0]
        noise_npy = np.load(file_path)
        best_noise = torch.from_numpy(noise_npy).cuda()
        print(file_path + " loaded")
    else:
        print("No matching file found.")

    global_model.eval()
    global_model = global_model.to(device)

    poisoned_workers = args.args_dict.fl_training.poisoned_workers
    target_label = args.args_dict.fl_training.target_label
    patch_mode = args.args_dict.narcissus_gen.patch_mode
    
    idx = poisoned_workers.index(client_idx) 
    target_class = target_label[idx]
    # poison_amount = round(poison_amount_ratio * args.args_dict.fl_training.n_target_samples[idx])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.49421428, 0.48513139, 0.45040909), (0.24665252, 0.24289226, 0.26159238)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    ori_test = torchvision.datasets.CIFAR10(root="./data/", train=False, download=False, transform=transform_test)
    
    multi_test = args.args_dict.narcissus_gen.multi_test
    criterion = nn.CrossEntropyLoss()

    poi_ori_test = torchvision.datasets.CIFAR10(root="./data/", train=False, download=False, transform=transform_test)
    
    # Attack success rate testing, estimated on test dataset, 10000 images of CIFAR-10
    test_label = [get_labels(ori_test)[x] for x in range(len(get_labels(ori_test)))]
    test_non_target = list(np.where(np.array(test_label)!=target_class)[0])
    test_non_target_change_image_label = poison_image_label(poi_ori_test, test_non_target, best_noise.cpu() * multi_test, target_class, None, patch_mode) # change original labels of poisoned inputs to the target label
    asr_loaders = torch.utils.data.DataLoader(test_non_target_change_image_label, batch_size=args.args_dict.fl_training.test_batch_size, shuffle=True, num_workers=0) # to compute the attack success rate (ASR)
    print('Poison test dataset size is:', len(test_non_target_change_image_label))

    #Clean acc test dataset
    clean_test_loader = torch.utils.data.DataLoader(ori_test, batch_size=args.args_dict.fl_training.test_batch_size, shuffle=False, num_workers=0)

    #Target clean test dataset
    test_target = list(np.where(np.array(test_label)==target_class)[0]) # grab test examples having label 2 (bird)
    target_test_set = Subset(ori_test, test_target) # create a subset of target class test examples in order to compute Tar-ACC
    target_test_loader = torch.utils.data.DataLoader(target_test_set, batch_size=args.args_dict.fl_training.test_batch_size, shuffle=True, num_workers=0) # to compute Tar-ACC

    correct, total = 0, 0
    for i, (images, labels) in enumerate(asr_loaders): # all examples labeled 2 (bird). Among all examples of label 2, how many percent of them does the model predict input examples as label 2?
        # 9000 samples from test dataset (with labels changed to target label)
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            logits = global_model(images)
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
            logits = global_model(images)
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
            logits = global_model(images)
            out_loss = criterion(logits, labels)
            _, predicted = torch.max(logits.data, 1)
            total_tar += labels.size(0)
            correct_tar += (predicted == labels).sum().item()
    acc_tar = correct_tar / total_tar
    print('\nTarget test clean Accuracy %.2f' % (acc_tar*100))
    print('Test_loss:', out_loss)


    args.get_logger().debug('Attack success rate: {}'.format(acc))
    args.get_logger().debug('\nTest clean Accuracy {}'.format(acc_clean))
    args.get_logger().debug('\nTarget test clean Accuracy {}'.format(acc_tar))
    
    print("\nFinish Server Testing ...\n" + " = "*16)
    return {"asr": acc, "clean_acc": acc_clean, "tar_acc": acc_tar}