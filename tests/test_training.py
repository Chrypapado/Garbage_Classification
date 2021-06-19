import pytest
import torch
from pathlib import Path
import sys
import os.path
project_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_dir / 'src/models'))
sys.path.insert(0, str(project_dir / 'src/data'))
import train_model  # noqa: E402
from make_dataset import main as dataset  # noqa: E402


def test_training():
    '''
    Test that makes sure that the training reports have been produced
    and that the trained model has been saved.
    '''
    report_path = str(project_dir / 'reports/figures/Training_loss.png')
    model_path = project_dir.joinpath('models/model.pth')

    assert os.path.isfile(report_path)
    assert os.path.isfile(model_path)
