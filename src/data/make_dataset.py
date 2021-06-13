# -*- coding: utf-8 -*-
import kaggle
import zipfile
import os
import shutil
import re
import logging
from pathlib import Path
from PIL import Image

import cv2
import kornia

import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    project_dir = Path(__file__).resolve().parents[2]
    parent_dir = project_dir.joinpath('./data/processed')
    train_path = os.path.join(parent_dir, 'Train')
    if os.path.exists(train_path):
        pass
    else:
        #Download Data
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('asdasdasasdas/garbage-classification', path=project_dir.joinpath('data/raw/'))

        #Unzip Files
        with zipfile.ZipFile(project_dir.joinpath('data/raw/garbage-classification.zip'), "r") as zip_ref:
            zip_ref.extractall(project_dir.joinpath('./data/processed'))

        #Create Files
        parent_dir = project_dir.joinpath('./data/processed')
        train = 'Train'
        test = 'Test'
        val = 'Validation'
        train_path = os.path.join(parent_dir, train)
        test_path = os.path.join(parent_dir, test)
        val_path = os.path.join(parent_dir, val)
        #if os.path.exists(train_path):
        #    pass
        #else:
        #    os.mkdir(train_path)
        #    os.mkdir(test_path)
        #    os.mkdir(val_path)
        os.mkdir(train_path)
        os.mkdir(test_path)
        os.mkdir(val_path)
        train_dir = project_dir.joinpath('./data/processed/Train')
        test_dir = project_dir.joinpath('./data/processed/Test')
        val_dir = project_dir.joinpath('./data/processed/Validation')
        classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        for i in range(len(classes)):
            train_path = os.path.join(train_dir, classes[i])
            test_path = os.path.join(test_dir, classes[i])
            val_path = os.path.join(val_dir, classes[i])
            #if os.path.exists(train_path):
            #    pass
            #else:
            #    os.mkdir(train_path)
            #    os.mkdir(test_path)
            #    os.mkdir(val_path)
            os.mkdir(train_path)
            os.mkdir(test_path)
            os.mkdir(val_path)

        #Split Data
        train_file = project_dir.joinpath('./data/processed/one-indexed-files-notrash_train.txt')
        test_file  = project_dir.joinpath('./data/processed/one-indexed-files-notrash_test.txt')
        val_file   = project_dir.joinpath('./data/processed/one-indexed-files-notrash_val.txt')
        df_train = pd.read_csv(train_file, sep=' ', header=None, names=['Path', 'Label'])
        df_test  = pd.read_csv(test_file,   sep=' ', header=None, names=['Path', 'Label'])
        df_val = pd.read_csv(val_file,   sep=' ', header=None, names=['Path', 'Label'])
        df_train['Path'] = df_train['Path'].apply(lambda x: x[:re.search("\d",x).start()] + '/' + x)
        df_test['Path'] = df_test['Path'].apply(lambda x: x[:re.search("\d",x).start()] + '/' + x)
        df_val['Path'] = df_val['Path'].apply(lambda x: x[:re.search("\d",x).start()] + '/' + x)
        for i in range(len(df_train)):
            source = str(project_dir.joinpath('./data/processed/Garbage classification/Garbage classification/' + df_train['Path'][i]))
            destination = str(project_dir.joinpath('./data/processed/Train/' + df_train['Path'][i].partition("/")[0]))
            print(source)
            if os.path.exists(destination + '/' + source.split('/')[-1]):
                pass
            else:
                shutil.move(source, destination)
        for i in range(len(df_test)):
            source = str(project_dir.joinpath('./data/processed/Garbage classification/Garbage classification/' + df_test['Path'][i]))
            destination = str(project_dir.joinpath('./data/processed/Test/' + df_train['Path'][i].partition("/")[0]))
            if os.path.exists(destination + '/' + source.split('/')[-1]):
                pass
            else:
                shutil.move(source, destination)
        for i in range(len(df_val)):
            source = str(project_dir.joinpath('./data/processed/Garbage classification/Garbage classification/' + df_val['Path'][i]))
            destination = str(project_dir.joinpath('./data/processed/Validation/' + df_train['Path'][i].partition("/")[0]))
            if os.path.exists(destination + '/' + source.split('/')[-1]):
                pass
            else:
                shutil.move(source, destination)

        #Generate Blur Images
        blur = list(df_train['Path'])[:len(list(df_train['Path']))//2]
        #Create the Gaussian Blur Filter
        gauss = kornia.filters.GaussianBlur2d((11, 11), (7, 7))
        for i in range(len(blur)):
            #Read the image with OpenCV
            img: np.ndarray = cv2.imread(str(project_dir.joinpath('./data/processed/Train/' + blur[i])))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #Convert to torch tensor
            data: torch.tensor = kornia.image_to_tensor(img, keepdim=False) 
            #Blur the image
            x_blur: torch.tensor = gauss(data.float())
            #Convert back to numpy
            img_blur: np.ndarray = kornia.tensor_to_image(x_blur.byte())
            #Save the Image
            im = Image.fromarray(img_blur)
            im.save(str(project_dir.joinpath('./data/processed/Train/' + blur[i].split('/')[0] + "/blur" + str(i) + ".jpg")))

        #Generate Augmented Images
        augment = list(df_train['Path'])[len(list(df_train['Path']))//2:]
        class MyAugmentation(nn.Module):
            def __init__(self):
                super(MyAugmentation, self).__init__()
                #We define and cache our operators as class members
                self.k1 = kornia.augmentation.ColorJitter(0.15, 0.25, 0.25, 0.25)
                self.k2 = kornia.augmentation.RandomAffine([-45., 45.], [0., 0.15], [0.5, 1.5], [0., 0.15])
            def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                img_out = self.k2(self.k1(img))
                mask_out = self.k2(mask, self.k2._params)
                return img_out, mask_out
        def load_data(data_path: str) -> torch.Tensor:
            data: np.ndarray = cv2.imread(data_path, cv2.IMREAD_COLOR)
            data_t: torch.Tensor = kornia.image_to_tensor(data, keepdim=False)
            data_t = kornia.bgr_to_rgb(data_t)
            data_t = kornia.normalize(data_t, 0., 255.)
            img, labels = data_t[..., :int(data_t.shape[-1]/2)], data_t[..., int(data_t.shape[-1]/2):]
            return img, labels
        for i in range(len(augment)):
            img, labels = load_data(str(project_dir.joinpath('./data/processed/Train/' + augment[i])))
            aug = MyAugmentation()
            img_aug, labels_aug = aug(img, labels)
            num_samples: int = 5
            for img_id in range(num_samples):
                img_aug, labels_aug = aug(img, labels)
                img_out = torch.cat([img_aug], dim=-1)
                plt.imsave(str(project_dir.joinpath('./data/processed/Train/' + augment[i].split('/')[0] + "/augment" + str(i) + f"_{img_id}.jpg")), kornia.tensor_to_image(img_out))

        #Delete Unnecessary Files
        shutil.rmtree(str(project_dir.joinpath('./data/processed/garbage classification')))
        shutil.rmtree(str(project_dir.joinpath('./data/processed/Garbage classification')))
        os.remove(str(project_dir.joinpath('./data/processed/one-indexed-files.txt')))
        os.remove(str(project_dir.joinpath('./data/processed/one-indexed-files-notrash_test.txt')))
        os.remove(str(project_dir.joinpath('./data/processed/one-indexed-files-notrash_train.txt')))
        os.remove(str(project_dir.joinpath('./data/processed//one-indexed-files-notrash_val.txt')))
        os.remove(str(project_dir.joinpath('./data/processed//zero-indexed-files.txt')))

    #Create Transform Object to Convert Data to Normalised Tensors
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    #Transform Data
    train_dir = str(project_dir.joinpath('./data/processed/Train'))
    trainset = ImageFolder(train_dir, transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    test_dir = str(project_dir.joinpath('./data/processed/Test'))
    testset = ImageFolder(test_dir, transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    val_dir = str(project_dir.joinpath('./data/processed/Validation'))
    valset = ImageFolder(val_dir, transform = transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

    return trainloader, testloader, valloader

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
