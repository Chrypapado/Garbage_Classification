import sys
from pathlib import Path

import pytest
import torch

project_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_dir / 'src/models'))
from model import ResNet  # noqa: E402


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_model():
    '''
    Check that given an input with shape 128
    that the output of the model has shape 10
    '''
    model = ResNet()
    X = 64
    y = model.forward(torch.rand(X, 3, 256, 256))

    assert y.shape == torch.Size([X, 6])
