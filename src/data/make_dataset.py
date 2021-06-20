import logging
import os
import re
import shutil
import zipfile
from pathlib import Path

import cv2
import kaggle
import kornia
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torchvision.datasets import ImageFolder

# import azureml.core
# from azureml.core import Workspace


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    # Load the workspace from the saved config file
    # ws = Workspace.from_config()
    # print('Ready to use Azure ML {} to work with {}'\
    # .format(azureml.core.VERSION, ws.name))

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    project_dir = str(Path(__file__).resolve().parents[2])
    processed_dir = project_dir + '/data/processed'
    train_dir = processed_dir + '/Train'
    test_dir = processed_dir + '/Test'
    val_dir = processed_dir + '/Validation'
    if os.path.exists(train_dir):
        pass
    else:
        # Download Data
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('asdasdasasdas/garbage-classification', path=project_dir + '/data/raw/')

        # Unzip Files
        with zipfile.ZipFile(project_dir + '/data/raw/garbage-classification.zip', "r") as zip_ref:
            zip_ref.extractall(processed_dir)
        # Creating Folders
        os.mkdir(train_dir)
        os.mkdir(test_dir)
        os.mkdir(val_dir)
        classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        for i in range(len(classes)):
            train_path = os.path.join(train_dir, classes[i])
            test_path = os.path.join(test_dir, classes[i])
            val_path = os.path.join(val_dir, classes[i])
            os.mkdir(train_path)
            os.mkdir(test_path)
            os.mkdir(val_path)

        # Split Data
        train_file = processed_dir + '/one-indexed-files-notrash_train.txt'
        test_file = processed_dir + '/one-indexed-files-notrash_test.txt'
        val_file = processed_dir + '/one-indexed-files-notrash_val.txt'
        df_train = pd.read_csv(train_file, sep=' ', header=None, names=['Path', 'Label'])
        df_test = pd.read_csv(test_file, sep=' ', header=None, names=['Path', 'Label'])
        df_val = pd.read_csv(val_file, sep=' ', header=None, names=['Path', 'Label'])
        df_train['Path'] = df_train['Path'].apply(lambda x: x[:re.search("\d", x).start()] + '/' + x)
        df_test['Path'] = df_test['Path'].apply(lambda x: x[:re.search("\d", x).start()] + '/' + x)
        df_val['Path'] = df_val['Path'].apply(lambda x: x[:re.search("\d", x).start()] + '/' + x)
        for i in range(len(df_train)):
            source = processed_dir + '/Garbage classification/Garbage classification/' + df_train['Path'][i]
            destination = train_dir + '/' + df_train['Path'][i].partition("/")[0]
            if os.path.exists(destination + '/' + source.split('/')[-1]):
                pass
            else:
                shutil.move(source, destination)
        for i in range(len(df_test)):
            source = processed_dir + '/Garbage classification/Garbage classification/' + df_test['Path'][i]
            destination = test_dir + '/' + df_test['Path'][i].partition("/")[0]
            if os.path.exists(destination + '/' + source.split('/')[-1]):
                pass
            else:
                shutil.move(source, destination)
        for i in range(len(df_val)):
            source = processed_dir + '/Garbage classification/Garbage classification/' + df_val['Path'][i]
            destination = val_dir + '/' + df_val['Path'][i].partition("/")[0]
            if os.path.exists(destination + '/' + source.split('/')[-1]):
                pass
            else:
                shutil.move(source, destination)

        # Generate Blur Images
        blur = list(df_train['Path'])[:len(list(df_train['Path'])) // 2]
        gauss = kornia.filters.GaussianBlur2d((11, 11), (7, 7))
        for i in range(len(blur)):
            img: np.ndarray = cv2.imread(train_dir + '/' + blur[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data: torch.tensor = kornia.image_to_tensor(img, keepdim=False)
            x_blur: torch.tensor = gauss(data.float())
            img_blur: np.ndarray = kornia.tensor_to_image(x_blur.byte())
            im = Image.fromarray(img_blur)
            im.save(train_dir + '/' + blur[i].split('/')[0] + "/blur" + str(i) + ".jpg")

        # Generate Augmented Images
        augment = list(df_train['Path'])[len(list(df_train['Path'])) // 2:]

        class MyAugmentation(nn.Module):

            def __init__(self):
                super(MyAugmentation, self).__init__()
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
            img, labels = data_t[..., :int(data_t.shape[-1])], data_t[..., int(data_t.shape[-1] / 2):]
            return img, labels

        for i in range(len(augment)):
            img, labels = load_data(train_dir + '/' + augment[i])
            aug = MyAugmentation()
            img_aug, labels_aug = aug(img, labels)
            num_samples: int = 5
            for img_id in range(num_samples):
                img_aug, labels_aug = aug(img, labels)
                img_out = torch.cat([img_aug], dim=-1)
                plt.imsave(train_dir + '/' + augment[i].split('/')[0] + "/augment" + str(i) + f"_{img_id}.jpg",
                           kornia.tensor_to_image(img_out))

        # Delete Unnecessary Files
        shutil.rmtree(processed_dir + '/garbage classification')
        shutil.rmtree(processed_dir + '/Garbage classification')
        os.remove(processed_dir + '/one-indexed-files.txt')
        os.remove(processed_dir + '/one-indexed-files-notrash_test.txt')
        os.remove(processed_dir + '/one-indexed-files-notrash_train.txt')
        os.remove(processed_dir + '/one-indexed-files-notrash_val.txt')
        os.remove(processed_dir + '/zero-indexed-files.txt')

    # Train, Test, Validation Datasets
    transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    train_set = ImageFolder(train_dir, transform=transformations)
    test_set = ImageFolder(test_dir, transform=transformations)
    val_set = ImageFolder(val_dir, transform=transformations)

    return train_set, test_set, val_set


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
