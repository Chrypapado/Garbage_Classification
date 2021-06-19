#Import Libraries and Modules
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from model import ResNet
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir / 'src/data'))
from make_dataset import main as dataset

class TrainOREvaluate(object):
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
        getattr(self, args.command)()

    def train(self):
        print("Training day and night")
        # Arguments
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=5.5e-5)
        parser.add_argument('--batch_size', default=64)
        parser.add_argument('--num_epochs', type=int, default=10)
        args = parser.parse_args(sys.argv[2:])
        print(args)
        # Load Data
        train_set, test_set, val_set = dataset()
        # Set Device and Model Configurations 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        model = ResNet()
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_dl = DataLoader(train_set, args.batch_size, shuffle = True, num_workers = 4, pin_memory = True)
        val_dl = DataLoader(val_set, args.batch_size * 2, num_workers = 4, pin_memory = True)
        # Training
        for epoch in range(args.num_epochs):
            print('Epoch '+ str(epoch + 1))
            model.train()
            for images, labels in train_dl:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            accuracy = 0
            counter = 0
            with torch.no_grad():
                model.eval()
                for images, labels in val_dl:
                    images, labels = images.to(device), labels.to(device)
                    ps = torch.exp(model(images))
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    counter += 1
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
                accuracy = accuracy / counter
                print('Accuracy: ' + str(accuracy*100) + '%')
            torch.save(model.state_dict(), 
                       project_dir.joinpath('models/model' + str(epoch) + '.pth'))

    def evaluate(self):
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
        test_dl = DataLoader(test_set, args.batch_size*2, num_workers = 4, pin_memory = True)
        print(device)
        dict_ = torch.load(model_path)
        model.load_state_dict(dict_)
        accuracy = 0
        counter = 0
        with torch.no_grad():
            model.eval()
            for images, labels in test_dl: 
                images, labels = images.to(device), labels.to(device)
                ps = torch.exp(model(images))
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                counter += 1
                accuracy += torch.mean(equals.type(torch.FloatTensor))
            accuracy = accuracy / counter
            print(f'Accuracy: {accuracy.item()*100}%')

if __name__ == '__main__':
    TrainOREvaluate()