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

from pathlib import Path
from tqdm import tqdm
from torchvision import datasets, transforms
from glob import glob

from torchvision import datasets, transforms

from parametricSN.data_loading.auto_augment import AutoAugment, Cutout
from parametricSN.data_loading.SmallSampleController import SmallSampleController

def imgtiny_augmentationFactory(augmentation, height, width):

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

def imgtiny_getDataloaders(trainSampleNum, valSampleNum, trainBatchSize, valBatchSize, trainAugmentation,
                       height, width, dataDir="."):
    # def cifar_getDataloaders(trainSampleNum, valSampleNum, trainBatchSize,

    datasetPath = '/xxx/ImageNet2012'
    print(datasetPath)
    
    if not os.path.isdir(datasetPath):
        print('Data Path Error!!!')


    transform_train = imgtiny_augmentationFactory(trainAugmentation, height, width)
    transform_val = imgtiny_augmentationFactory('noaugment', height, width)

    dataset_train = datasets.ImageFolder(root=os.path.join(datasetPath, 'train'),  # use train dataset
                                         transform=transform_train)

    dataset_val = datasets.ImageFolder(root=os.path.join(datasetPath, 'val'),  # use train dataset
                                       transform=transform_val)
    ssc = SmallSampleController(
        trainSampleNum=trainSampleNum, valSampleNum=valSampleNum,
        trainBatchSize=trainBatchSize, valBatchSize=valBatchSize,
        trainDataset=dataset_train, valDataset=dataset_val
    )

    return ssc
