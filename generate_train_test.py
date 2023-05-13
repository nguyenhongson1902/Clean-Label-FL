from torchvision import datasets, transforms
import torchvision
import torch


def get_dataset(args, kwargs):
	if args.dataset == "EMNIST":
		train_dataset = datasets.EMNIST(
			"./data",
			split="digits",
			train=True,
			download=True,
			transform=transforms.Compose(
				[transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
			),
		)
		test_dataset = datasets.EMNIST(
			"./data",
			split="digits",
			train=False,
			transform=transforms.Compose(
				[transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
			),
		)
	elif args.dataset == "MNIST":
		train_dataset = datasets.MNIST(
			"./data",
			train=True,
			download=True,
			transform=transforms.Compose(
				[transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
			),
		)
		test_dataset = datasets.MNIST(
			"./data",
			train=False,
			transform=transforms.Compose(
				[transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
			),
		)

	elif args.dataset == "Cifar10":
		transform_train = transforms.Compose(
			[
				transforms.RandomCrop(32, padding=4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(
					# (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
					# (0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)
					(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
				),
			]
		)

		transform_test = transforms.Compose(
			[
				transforms.ToTensor(),
				transforms.Normalize(
					# (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
					# (0.49421428, 0.48513139, 0.45040909), (0.24665252, 0.24289226, 0.26159238)
					(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
				),
			]
		)

		train_dataset = torchvision.datasets.CIFAR10(
			root="./data", train=True, download=True, transform=transform_train
		)

		test_dataset = torchvision.datasets.CIFAR10(
			root="./data", train=False, download=True, transform=transform_test
		)

	elif args.dataset == "TinyImageNet":
		train_dataset = torchvision.datasets.ImageFolder(
			root="./data/tiny-imagenet-200/train",
			transform=transforms.Compose(
				[
					transforms.RandomResizedCrop(64),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					transforms.Normalize(
						mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
					),
				]
			),
		)

		test_dataset = torchvision.datasets.ImageFolder(
			root="./data/tiny-imagenet-200/val",
			transform=transforms.Compose(
				[
					transforms.Resize(64),
					transforms.CenterCrop(64),
					transforms.ToTensor(),
					transforms.Normalize(
						mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
					),
				]
			),
		)

	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
	)

	test_loader = torch.utils.data.DataLoader(
		test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs
	)

	return train_dataset, test_dataset, train_loader, test_loader

