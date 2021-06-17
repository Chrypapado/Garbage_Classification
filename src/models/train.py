import sys
import argparse
from pathlib import Path

from torch.utils.data.dataloader import DataLoader

from model import ImageClassificationBase
from model import ResNet
from device import get_default_device, to_device, DeviceDataLoader

import torch

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir / 'src/data'))
from make_dataset import main as dataset

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python train_model.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        #Use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=5.5e-5)
        parser.add_argument('--batch_size', default=64)
        parser.add_argument('--num_epochs', default=10)
        args = parser.parse_args(sys.argv[2:])
        print(args)
        train_set, test_set, val_set = dataset()
        model = ResNet()
        #train_dl = DataLoader(train_set, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
        #val_dl = DataLoader(val_set, batch_size*2, num_workers = 4, pin_memory = True)
        train_dl = DataLoader(train_set, args.batch_size)
        val_dl = DataLoader(val_set, args.batch_size*2)
        device = get_default_device()
        train_dl = DeviceDataLoader(train_dl, device)
        val_dl = DeviceDataLoader(val_dl, device)
        to_device(model, device)
        model = to_device(ResNet(), device)
        history = []
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(args.num_epochs):
            model.train()
            train_losses = []
            for batch in train_dl:
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            with torch.no_grad():
                model.eval()
                outputs = [model.validation_step(batch) for batch in val_dl]
                result = model.validation_epoch_end(outputs)
                result['train_loss'] = torch.stack(train_losses).mean().item()
                model.epoch_end(epoch, result)
            torch.save(model.state_dict(), project_dir.joinpath('models/model' + str(epoch) + '.pth'))

if __name__ == '__main__':
    TrainOREvaluate()