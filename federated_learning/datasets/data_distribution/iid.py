import torch
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.utils.data import Subset
# from torch.utils.data import DataLoader
import numpy as np
import random

np.random.seed(1)
random.seed(1)


def generate_iid_data(train_dataset, test_dataset, args, kwargs):
	N = args.num_clients  # number of clients in total
	all_range = list(range(len(train_dataset)))
	random.shuffle(all_range)
	print("all_range: ", all_range)
	data_len_per_client = len(train_dataset) // N
	# print("data_len_per_client: ", data_len_per_client)
	train_indices = [
		all_range[i * data_len_per_client : (i + 1) * data_len_per_client]
		for i in range(N)
	]
	# train_loaders = [
	# 	torch.utils.data.DataLoader(
	# 		train_dataset,
	# 		batch_size=args.batch_size,
	# 		sampler=SubsetRandomSampler(indices),
	# 		**kwargs
	# 	)
	# 	for indices in train_indices
	# ]

	train_loaders = [
    torch.utils.data.DataLoader(
        Subset(train_dataset, indices),  # Use Subset to create a subset of the dataset
        batch_size=args.batch_size,
		shuffle=True,
        **kwargs
    )
    for indices in train_indices
	]

	test_loader = torch.utils.data.DataLoader(
		test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs
	)

	return train_loaders, train_indices, test_loader


