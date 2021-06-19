import pytest
import torch
from pathlib import Path
import sys
project_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_dir / 'src/models'))
from model import MyAwesomeModel  # noqa: E402


def test_model():
    '''
    Check that given an input with shape 128
    that the output of the model has shape 10
    '''
    model = MyAwesomeModel()
    X = 128
    y = model.forward(torch.rand(X, 1, 28, 28))

    assert y.shape == torch.Size([X, 10])
