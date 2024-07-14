"""Contains classes and functions for downloading and sampling the ImageNet/ImageNet-Tiny dataset

Author: Dawn

Dataset folder organization
├── a    <- Folder contains 11 folders (one per material)   
├── b    <- Folder contains 11 folders (one per material) 
├── c    <- Folder contains 11 folders (one per material) 
├── d    <- Folder contains 11 folders (one per material)    

Functions: 
    img_augmentationFactory -- factory of ImageNet augmentations
    img_getDataloaders      -- returns dataloaders for ImageNet
    download_from_url       -- Download Dataset from URL
    extract_tar             -- Extract files from tar
    create_dataset          -- Create dataset in the target folder
    downloadImageNet      -- Download ImageNet dataset to the data folder

class:
    ImgLoader -- loads ImageNet from disk and creates dataloaders
"""

import shutil
import os
import requests
import tarfile
import sys
import torch
import time
import shutil

from pathlib import Path
from tqdm import tqdm
from torchvision import datasets, transforms
from glob import glob

# from parametricSN.data_loading.auto_augment import AutoAugment, Cutout

from torchvision import datasets, transforms

from parametricSN.data_loading.auto_augment import AutoAugment, Cutout
from parametricSN.data_loading.SmallSampleController import SmallSampleController



def img_augmentationFactory(augmentation, height, width):
    """Factory for different augmentation choices for Imagenet"""

    if augmentation == 'autoaugment':
        transform = [
            transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            Cutout()
        ]

    elif augmentation == 'original-cifar':
        transform = [
            transforms.Resize((200,200)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip(),
        ]

    elif augmentation == 'noaugment':
        transform = [
            transforms.Resize((200, 200)),
            transforms.CenterCrop((height, width))
        ]

    elif augmentation == 'glico':
        NotImplemented(f"augment parameter {augmentation} not implemented")

    else: 
        NotImplemented(f"augment parameter {augmentation} not implemented")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return transforms.Compose(transform + [transforms.ToTensor(), normalize])



# def img_getDataloaders(trainBatchSize, valBatchSize, trainAugmentation,
                       # height, width, sample, dataDir="."):
def img_getDataloaders(trainSampleNum, valSampleNum, trainBatchSize, valBatchSize, trainAugmentation,
                       height, width, sample, dataDir="."):
    # def cifar_getDataloaders(trainSampleNum, valSampleNum, trainBatchSize,
    #                          valBatchSize, trainAugmentation, dataDir="."):
    """Samples a specified class balanced number of samples form the KTH-TIPS2 dataset
    
    returns:
        loader
    """
    datasetPath = 'xxx/ImageNet2012'

    print(datasetPath)
    
    if not os.path.isdir(datasetPath):
        # downloadKTH_TIPS2()
        print('Data Path Error!!!')


    transform_train = img_augmentationFactory(trainAugmentation, height, width)
    transform_val = img_augmentationFactory('noaugment', height, width)

    dataset_train = datasets.ImageNet(#load train dataset
        root=dataDir, train=True,
        transform=transform_train, download=False
    )

    dataset_val = datasets.ImageNet(#load test dataset
        root=dataDir, train=False,
        transform=transform_val, download=False
    )

    ssc = SmallSampleController(
        trainSampleNum=trainSampleNum, valSampleNum=valSampleNum,
        trainBatchSize=trainBatchSize, valBatchSize=valBatchSize,
        trainDataset=dataset_train, valDataset=dataset_val
    )
    return ssc #loader

class ImgLoader():

    def __init__(self, data_dir, train_batch_size, val_batch_size, 
                 transform_train, transform_val, sample='a'):

        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.transform_train = transform_train
        self.transform_val = transform_val
        self.sample = sample

    def generateNewSet(self, device, workers=5, seed=None, load=False):
        datasets_val = []
        for s in ['a', 'b', 'c', 'd']:
            if self.sample == s:
                dataset = datasets.ImageFolder(#load train dataset
                    root=os.path.join(self.data_dir,s), 
                    transform=self.transform_train
                )
                dataset_train = dataset
            else:
                dataset = datasets.ImageFolder(#load train dataset
                    root=os.path.join(self.data_dir,s), 
                    transform=self.transform_val
                )

                datasets_val.append(dataset)

        dataset_val = torch.utils.data.ConcatDataset(datasets_val)

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=self.train_batch_size, 
                                                   shuffle=True, num_workers=workers,
                                                   pin_memory=True)

        test_loader = torch.utils.data.DataLoader(dataset_val, batch_size=self.val_batch_size, 
                                                  shuffle=True, num_workers=workers, 
                                                  pin_memory=True)

        self.trainSampleCount, self.valSampleCount = sum([len(x) for x in train_loader]), sum([len(x) for x in test_loader])

        if load:
            for batch,target in train_loader:
                batch.cuda()
                target.cuda()

            for batch,target in test_loader:
                batch.cuda()
                target.cuda()    

        if seed == None:
            seed = int(time.time()) #generate random seed
        else:
            seed = seed

        return train_loader, test_loader, seed




def create_dataset(target_path):
    """Create KTH dataset in the target folder
    Parameters:
        target_path -- path to the new dataset folder
    """
    folders = glob(f'{target_path}/KTH-TIPS2-b/*/*')
    print("Creating new dataset folder")
    for folder in tqdm(folders):
        new_folder = os.path.join(target_path, "KTH")
        sample = folder.split('/')[-1][-1]
        label = folder.split('/')[-2]
        destination_path = os.path.join(new_folder, f'{sample}/{label}')
        print(destination_path)
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)   
        pattern = f'{folder}*/*' 
        for img in glob(pattern):
            shutil.copy(img, destination_path)

