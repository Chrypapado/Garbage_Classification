# Import Libraries and Modules
import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data.dataloader import DataLoader
from model import ResNet

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir / 'src/data'))
from make_dataset import main as dataset  # noqa: E402


def evaluate():
    print("Evaluating until hitting the ceiling")
    # Arguments
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--load_model_from', default='model0')
    parser.add_argument('--batch_size', default=64)
    args = parser.parse_args(sys.argv[2:])
    print(args)
    # Load Model
    model_path = str(project_dir.joinpath('./models')) + '/' + args.load_model_from + '.pth'
    # Load Data
    train_set, test_set, val_set = dataset()
    # Set Device and Model Configurations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = ResNet()
    model.to(device)
    test_dl = DataLoader(test_set, args.batch_size * 2, shuffle=True, num_workers=4, pin_memory=True)
    if torch.cuda.is_available():
        dict_ = torch.load(model_path)
    else:
        dict_ = torch.load(model_path, map_location='cpu')
    model.load_state_dict(dict_)
    with torch.no_grad():
        test_accuracy = 0
        model.eval()
        for (counter, (images, labels)) in enumerate(test_dl):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, dim=1)
            test_acc = torch.tensor(torch.sum(predictions == labels).item() / len(predictions))
            test_accuracy += test_acc
        accuracy = test_accuracy.item() / (counter + 1)
        print('Test accuracy: {:.2f}%'.format(accuracy * 100))


if __name__ == '__main__':
    evaluate()
