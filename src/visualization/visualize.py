#Import Libraries and Modules
import sys
import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir / 'src/data'))
from make_dataset import main as dataset
sys.path.insert(0, str(project_dir / 'src/models'))
from model import ResNet

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--load_model_from', default='model0')
    args = parser.parse_args(sys.argv[2:])
    print(args)
    # Load Model
    model_path = str(project_dir.joinpath('./models')) + '/' + args.load_model_from + '.pth'
    # Load Data
    train_set, test_set, val_set = dataset()
    val_dl = DataLoader(val_set, args.batch_size, num_workers = 4, pin_memory = True, shuffle = True)    
    # Set Device and Model Configurations 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = ResNet()
    model.to(device)
    dict_ = torch.load(model_path)
    model.load_state_dict(dict_)
    # Features
    features_1 = nn.Sequential(*list(model.children()))
    features_2 = nn.Sequential(*list(features_1[0].children()))
    features_3 = features_2[:-1]
    for images, labels in val_dl:
        images, labels = images.to(device), labels.to(device)
        features_4 = features_3(images)
        break
    features_5 = features_4.detach().squeeze().cpu().numpy()
    embedded = TSNE(n_components=2).fit_transform(features_5)
    classes = pd.Series(list(labels.cpu().numpy())).map({0:'Glass', 1:'Paper', 2:'Cardboard', 3:'Plastic', 4:'Metal', 5:'Trash'})
    data = pd.DataFrame({'x': embedded[:, 0], 'y': embedded[:, 1], 'label': classes})
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=data, x='x', y='y', hue='label', ax=ax, palette="Set1")
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.title('t-SNE embedding of the features')
    plt.savefig(str(project_dir.joinpath('./reports/figures/tSNE.png')))