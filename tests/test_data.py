import sys
from pathlib import Path

import pytest
import torch

project_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_dir / 'src/data'))
from make_dataset import main as dataset  # noqa: E402


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_data():
    '''
    Checks that data gets loaded correctly
    '''
    trainset, testset, valset = dataset()
    assert len(trainset) == 7072
    assert len(testset) == 431
    assert len(valset) == 328
    trainset = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True)
    image, label = next(iter(trainset))
    assert image[0].shape == torch.Size([3, 256, 256])
    # make sure that all labels are represented
    label_list = label.tolist()
    labels = torch.tensor([0, 1, 2, 3, 4, 5])
    labels = labels.tolist()
    result = all(item in label_list for item in labels)
    assert result == True
