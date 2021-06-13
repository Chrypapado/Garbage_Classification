import argparse
import sys
from pathlib import Path

import torch

import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt

from model import MyAwesomeModel

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir / 'src/data'))
from make_dataset import main as dataset

if __name__ == '__main__':
    train_set, test_set, val_set = dataset()
    model = MyAwesomeModel()
    dict_ = torch.load(project_dir.joinpath('models/model.pth'))
    model.load_state_dict(dict_)
    inputs, classes = next(iter(val_set))
    outputs = model(inputs)
    ps = torch.exp(outputs)
    predictions = ps.max(1)[1]
    #Show Image
    plt.imshow(inputs[0].permute(1, 2, 0))
    plt.show()
    clas = pd.Series(list(classes.numpy())).map({0:'Glass', 1:'Paper', 2:'Cardboard', 3:'Plastic', 4:'Metal', 5:'Trash'})[0]
    pred = pd.Series(list(predictions.numpy())).map({0:'Glass', 1:'Paper', 2:'Cardboard', 3:'Plastic', 4:'Metal', 5:'Trash'})[0]
    print('Label: ', clas, ', Predicted: ', pred)