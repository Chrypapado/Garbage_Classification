import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import time
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


class GarbageDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        self.path_to_folder = path_to_folder
        self.transform = transform
        self.labels, self.image_paths = [], []
        classes = os.listdir(self.path_to_folder)
        for label in classes:
            path = os.path.join(self.path_to_folder, label)
            images = os.listdir(path)
            for image in images:
                self.labels.append(label)
                self.image_paths.append(os.path.join(path, image))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image_trans = self.transform(image)
        label = self.labels[index]
        return image_trans, label


def loader(path, batch_size, num_workers):
    transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    dataset = GarbageDataset(path, transformations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dataloader


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--path', type=str, default='Train')
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args(sys.argv[2:])
    print(args)
    # Define Path
    project_dir = str(Path(__file__).resolve().parents[2])
    path = project_dir + '/data/processed/' + args.path
    # Dataloader
    dataloader = loader(path, args.batch_size, args.num_workers)
    # Visualize Batch
    images, labels = next(iter(dataloader))
    plt.figure(figsize=(20, 10))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(np.swapaxes(np.swapaxes(images[i].numpy(), 0, 2), 0, 1))
        plt.title(labels[i])
        plt.axis('off')
    plt.show()
    # Errorbar plot
    means, stds = [], []
    for num_workers in range(4):
        res = []
        dataloader = loader(path, args.batch_size, num_workers + 1)
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > 100:
                    break
            end = time.time()
            res.append(end - start)
        res = np.array(res)
        means += [np.mean(res)]
        stds += [np.std(res)]
    plt.figure(figsize=(12, 8))
    plt.errorbar(list(range(1, 5)), means, yerr=stds)
    plt.xlabel('Number of Workers')
    plt.ylabel('Time')
    plt.savefig(project_dir + '/reports/figures/errorbar.png')
