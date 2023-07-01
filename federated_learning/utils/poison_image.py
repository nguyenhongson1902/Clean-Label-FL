from .apply_noise_patch import apply_noise_patch
import torch
from torch.utils.data import Dataset


class poison_image(Dataset):
    def __init__(self, dataset, indices, noise, transform, patch_mode):
        """
        The function initializes an object with the given dataset, indices, noise, transform, and patch
        mode.
        
        :param dataset: The dataset parameter is the input dataset that you want to apply
        transformations to. It could be a list, array, or any other data structure that represents your
        dataset
        :param indices: The `indices` parameter is a list or array of indices that specify which samples
        from the dataset should be used. These indices can be used to select a subset of the dataset for
        training or testing purposes
        :param noise: The "noise" parameter is used to control the amount of noise that is added to the
        dataset. It can be a value between 0 and 1, where 0 means no noise is added and 1 means maximum
        noise is added
        :param transform: The `transform` parameter is used to apply transformations to the data. It can
        be a function or an object that implements the `__call__` method. This allows you to perform
        various operations on the data, such as resizing, cropping, or applying data augmentation
        techniques. The transformed data will be
        :param patch_mode: The `patch_mode` parameter is used to specify how the dataset should be
        divided into patches. It determines the size and arrangement of the patches within the dataset
        """
        self.dataset = dataset
        self.indices = indices
        self.noise = noise
        self.transform = transform
        self.patch_mode = patch_mode

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        if idx in self.indices:
            # image = torch.clamp(apply_noise_patch(self.noise,image,mode="add"),-1,1)
            image = torch.clamp(apply_noise_patch(self.noise, image, mode=self.patch_mode), -1, 1)
        label = self.dataset[idx][1]
        if self.transform is not None:
            image = self.transform(image)
        return (image, label)

    def __len__(self):
        return len(self.dataset)