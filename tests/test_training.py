import os.path
import sys
from pathlib import Path

import pytest

project_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_dir / 'src/models'))
sys.path.insert(0, str(project_dir / 'src/data'))


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_training():
    '''
    Test that makes sure that the t-SNE figure has been produced
    and that the trained model has been saved.
    '''
    report_path = str(project_dir / 'reports/figures/tSNE.png')
    model_path = project_dir.joinpath('models/model0.pth')

    assert os.path.isfile(report_path)
    assert os.path.isfile(model_path)
