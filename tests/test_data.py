import torch
import pytest
from pathlib import Path
import sys
project_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_dir / 'src/data'))
from make_dataset import main as dataset # noqa: E402


def test_data():
    '''
    Checks that data gets loaded correctly
    '''
    trainset, testset = dataset()
    assert len(trainset) == 60000
    assert len(testset) == 10000
    trainset = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True)
    image, label = next(iter(trainset))
    # image[0].shapge gives me an error I do not know why
    assert image[0].shape == torch.Size([1, 28, 28])
    # make sure that all labels are represented
    label_list = label.tolist()
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    labels = labels.tolist()
    result = all(item in label_list for item in labels)
    assert result == True
