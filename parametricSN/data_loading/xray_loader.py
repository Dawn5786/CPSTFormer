import os
import shutil

import pandas as pd

from pathlib import Path
from zipfile import ZipFile
from tqdm import tqdm
from torchvision import datasets, transforms

from parametricSN.data_loading.auto_augment import AutoAugment, Cutout
from parametricSN.data_loading.SmallSampleController import SmallSampleController


def xray_augmentationFactory(augmentation, height, width):

    downsample = (260,260)

    if augmentation == 'autoaugment':
        transform = [
            transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            Cutout()
        ]
    elif augmentation == 'original-cifar':
        transform = [
            transforms.Resize(downsample),
            transforms.RandomCrop(size=(height, width)),
            transforms.RandomHorizontalFlip(),
        ]
    elif augmentation == 'noaugment':
        transform = [
            transforms.Resize(downsample),
            transforms.CenterCrop((height, width)),
        ]

    elif augmentation == 'glico':
        NotImplemented(f"augment parameter {augmentation} not implemented")
    else: 
        NotImplemented(f"augment parameter {augmentation} not implemented")

    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     #std=[0.229, 0.224, 0.225])

    normalize = transforms.Normalize(mean=[0.5888, 0.5888, 0.5889],
                                     std=[0.1882, 0.1882, 0.1882])

    return transforms.Compose(transform + [transforms.ToTensor(), normalize])

def xray_getDataloaders(trainSampleNum, valSampleNum, trainBatchSize, 
                        valBatchSize, trainAugmentation, height, 
                        width, dataDir="."):


    if not os.path.isdir(Path(os.path.realpath(__file__)).parent.parent.parent/'data'/'xray_preprocess'):
        downloadCOVIDXCRX2()
    
    transform_train = xray_augmentationFactory(trainAugmentation, height, width)
    transform_val = xray_augmentationFactory("noaugment", height, width)

    dataset_train= datasets.ImageFolder(root=os.path.join(dataDir,'train'), #use train dataset
                                            transform=transform_train)

    dataset_val = datasets.ImageFolder(root=os.path.join(dataDir,'test'), #use train dataset
                                            transform=transform_val)

    ssc = SmallSampleController(
        trainSampleNum=trainSampleNum, valSampleNum=valSampleNum, 
        trainBatchSize=trainBatchSize, valBatchSize=valBatchSize, 
        trainDataset=dataset_train, valDataset=dataset_val
    )

    return ssc

def extract_zip(dataset_path, target_path):

    dataset_path = os.path.join(dataset_path,'covidx-cxr2.zip')
    print(f'Extracting zip file: {dataset_path}')
    with ZipFile(file=dataset_path) as zip_file:
        zip_file.extractall(path=os.path.join(target_path, 'xray'))
    os.remove(dataset_path)

def create_train_folder(df_train, target_path):
  
    folder_path = os.path.join(target_path, 'xray_preprocess/train')
    print(f'Create train set at: {folder_path}')
    for _, row in tqdm(df_train.iterrows(), total=df_train.shape[0]):
        if row['class']=='negative':
            destination_path = os.path.join(folder_path, 'negative')
        elif row['class']=='positive':
            destination_path = os.path.join(folder_path, 'positive')
        if not os.path.exists(destination_path):
            os.makedirs(destination_path) 
        img = os.path.join(target_path, 'xray', 'train', row['filename'])
        shutil.copy(img, destination_path )

def create_test_folder(df_test, target_path):
 
    folder_path = os.path.join(target_path, 'xray_preprocess/test')
    print(f'Create test set at: {folder_path}')
    for _, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
        if row['class']=='negative':
            destination_path = os.path.join(folder_path, 'negative')
        elif row['class']=='positive':
            destination_path = os.path.join(folder_path, 'positive')
        if not os.path.exists(destination_path):
            os.makedirs(destination_path) 
        img = os.path.join(target_path, 'xray', 'test', row['filename'])
        shutil.copy(img, destination_path )

def create_train_test_df(target_path):
 
    df_train = pd.read_csv(os.path.join(target_path, 'xray', 'train.txt'), delimiter=' ',
                                        header = 0 )
    df_test = pd.read_csv(os.path.join(target_path, 'xray', 'test.txt'), delimiter=' ', header = 0)
    df_train.columns=['patient_id', 'filename', 'class', 'data_source']
    df_test.columns=['patient_id', 'filename', 'class', 'data_source']

    return df_train, df_test


def downloadCOVIDXCRX():
    target_path = Path(os.path.realpath(__file__)).parent.parent.parent/'data'
    target_path.mkdir(parents=True,exist_ok=True)

    os.system('kaggle datasets download -d andyczhao/covidx-cxr2 --force')
    cwd = Path(os.path.realpath(__file__)).parent.parent.parent

    extract_zip(cwd, target_path)
    df_train, df_test = create_train_test_df(target_path)
    create_train_folder(df_train, target_path)
    create_test_folder(df_test, target_path)
    mydir = os.path.join(target_path,'xray')
    try:
        shutil.rmtree( mydir)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
