import torch


class concate_dataset(torch.utils.data.Dataset):
    def __init__(self, target_dataset,outter_dataset):
        """
        self.idataset: in-distribution dataset
        self.odataset: out-distribution dataset
        """
        self.idataset = target_dataset # note that target_dataset is from CIFAR-10, 5000 images of bird
        self.odataset = outter_dataset # outter_trainset is from Tiny ImageNet (POOD dataset)

    def __getitem__(self, idx):
        if idx < len(self.odataset):
            img = self.odataset[idx][0]
            labels = self.odataset[idx][1]
        else:
            img = self.idataset[idx-len(self.odataset)][0]
            #labels = torch.tensor(len(self.odataset.classes),dtype=torch.long)
            labels = len(self.odataset.classes)
        #label = self.dataset[idx][1]
        return (img,labels)

    def __len__(self):
        return len(self.idataset)+len(self.odataset)