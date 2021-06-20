# Import Libraries and Modules
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from model import ResNet
from torch.utils.data.dataloader import DataLoader

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir / 'src/data'))
import wandb  # noqa: E402
from make_dataset import main as dataset  # noqa: E402

wandb.login()


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
        wandb.init(project='testing', config=args)
        # Load Data
        train_set, test_set, val_set = dataset()
        # Set Device and Model Configurations
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        model = ResNet()
        model.to(device)
        wandb.watch(model, log_freq=100)
        classes = ['Glass', 'Paper', 'Cardboard',
                   'Plastic', 'Metal', 'Trash']
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_dl = DataLoader(train_set, args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
        val_dl = DataLoader(val_set, args.batch_size * 2,
                            num_workers=4, pin_memory=True)
        # Training
        for epoch in range(args.num_epochs):
            print('Epoch ' + str(epoch + 1))
            model.train()
            class_acc = {}
            correct_pred_list = [[0] for _ in range(6)]
            total_pred_list = [[0] for _ in range(6)]
            for images, labels in train_dl:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                wandb.log({"Training Loss": loss})
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            accuracy = 0
            counter = 0
            with torch.no_grad():
                model.eval()
                for images, labels in val_dl:
                    # val = model.validation_step(batch)
                    # wandb.log({"Validation Accuracy": val['val_acc']})
                    # wandb.log({"Validation Loss": val['val_loss']})
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    val_loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, dim=1)
                    val_acc = torch.tensor(
                        torch.sum(preds == labels).item() / len(preds))
                    wandb.log({"Validation Accuracy": val_acc})
                    wandb.log({"Validation Loss": val_loss})
                    ps = torch.exp(outputs)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    _, predictions = torch.max(outputs, 1)
                    counter += 1
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

                    for label, prediction in zip(labels, predictions):
                        if label == prediction:
                            correct_pred_list[label][0] += 1
                        # correct_pred[classes[label]] += 1
                        total_pred_list[label][0] += 1

                accuracy = accuracy / counter
                print('Accuracy: ' + str(accuracy * 100) + '%')
            for i in range(len(classes)):

                if total_pred_list[i][0] != 0:
                    print('Accuracy of %5s : %2d %%' %
                          (classes[i], 100 * correct_pred_list[i][0] / total_pred_list[i][0]))
                    class_acc["Accuracy of %5s" %
                              (classes[i])] = 100 * correct_pred_list[i][0] / total_pred_list[i][0]
                else:
                    print('Accuracy of %5s : %2d %%' %
                          (classes[i], 100 * correct_pred_list[i][0] / 1))
                    class_acc["Accuracy of %5s" %
                              (classes[i])] = 100 * correct_pred_list[i][0] / 1
            wandb.log(class_acc)

            torch.save(model.state_dict(),
                       project_dir.joinpath('models/model' + str(epoch) + '.pth'))
        wandb.finish()

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        # Arguments
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default='model0')
        parser.add_argument('--batch_size', default=64)
        args = parser.parse_args(sys.argv[2:])
        print(args)
        # Load Model
        model_path = str(project_dir.joinpath('./models')) + \
            '/' + args.load_model_from + '.pth'
        # Load Data
        train_set, test_set, val_set = dataset()
        # Set Device and Model Configurations
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        model = ResNet()
        model.to(device)
        test_dl = DataLoader(test_set, args.batch_size * 2,
                             num_workers=4, pin_memory=True)
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
