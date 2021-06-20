import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from model import ResNet
from PIL import Image

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--image', default='example.jpg')
    parser.add_argument('--load_model_from', default='model0')
    args = parser.parse_args(sys.argv[2:])
    print(args)
    # Load Model
    project_dir = Path(__file__).resolve().parents[2]
    model_path = str(project_dir.joinpath('./models')) + \
        '/' + args.load_model_from + '.pth'
    # Set Device and Model Configurations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = ResNet()
    model.to(device)
    dict_ = torch.load(model_path, map_location='cpu')
    model.load_state_dict(dict_)
    # Image Settings
    image = Image.open(str(project_dir.joinpath(
        './data/external')) + '/' + args.image)
    transformations = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()])
    edit_image = transformations(image)
    plt.imshow(edit_image.permute(1, 2, 0))
    plt.show()
    unsqueezed_image = edit_image.unsqueeze(0)
    device_image = unsqueezed_image.to(device, non_blocking=True)
    modeled_image = model(device_image)
    prob, preds = torch.max(modeled_image, dim=1)
    # Prediction
    classes = ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal', 'Trash']
    classes[preds[0].item()]
    print("Predicted image: ", classes[preds[0].item()])
