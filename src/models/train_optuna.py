# Import Libraries and Modules
import sys
import argparse
from pathlib import Path
import wandb
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from model import ResNet
from torchvision.datasets import ImageFolder
import optuna
# from optuna.visualization import plot_optimization_history
# from optuna.visualization import plot_intermediate_values
import os

if not os.path.exists("Images"):
    os.mkdir("Images")

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir / 'src/data'))
from make_dataset import main as dataset  # noqa: E402


def objective(trial):
    project_dir = str(Path(__file__).resolve().parents[2])
    processed_dir = project_dir + '/data/processed'
    train_dir = processed_dir + '/Train'
    # test_dir = processed_dir + '/Test'
    val_dir = processed_dir + '/Validation'

    lr = trial.suggest_discrete_uniform('lr', 1e-6, 1e-0, 1e-6)
    batch_size = int(trial.suggest_discrete_uniform('batch_size', 5, 30, 5))
    # batch_normalize = trial.suggest_categorical('batch_normalize', ['True', 'False'])
    # activation_function = trial.suggest_categorical('activations', [nn.ReLU, nn.Tanh, nn.RReLU, nn.LeakyReLU, nn.ELU])

    # if batch_normalize == False:
    transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    train_set = ImageFolder(train_dir, transform=transformations)
    # test_set = ImageFolder(test_dir, transform=transformations)
    val_set = ImageFolder(val_dir, transform=transformations)
    # else:
    #     transformations = transforms.Normalize([transforms.Resize((256, 256)), transforms.ToTensor()])
    #     train_set = ImageFolder(train_dir, transform=transformations)
    #     test_set = ImageFolder(test_dir, transform=transformations)
    #     val_set = ImageFolder(val_dir, transform=transformations)

    # DATALOADERS
    train_dl = DataLoader(train_set, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_set, batch_size * 2, shuffle=True, num_workers=4, pin_memory=True)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = ResNet()
    model.to(device)
    print(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = 10
    for epoch in range(epochs):

        train_accuracy = 0
        model.train()

        for (counter, (images, labels)) in enumerate(train_dl):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predictions = torch.max(output, dim=1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_acc = torch.tensor(torch.sum(predictions == labels).item() / len(predictions))
            train_accuracy += train_acc
        # epoch_train_acc = train_accuracy.item() / (counter + 1)

        with torch.no_grad():
            val_accuracy = 0
            # class_acc = {}
            # correct_pred_list = [0 for _ in range(6)]
            # total_pred_list = [0 for _ in range(6)]
            model.eval()
            for (counter, (images, labels)) in enumerate(val_dl):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predictions = torch.max(outputs, dim=1)
                val_acc = torch.tensor(torch.sum(predictions == labels).item() / len(predictions))
                val_accuracy += val_acc

            epoch_val_acc = val_accuracy.item() / (counter + 1)

    return epoch_val_acc


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
        parser.add_argument('--batch_size', default=32)
        parser.add_argument('--num_epochs', type=int, default=10)
        args = parser.parse_args(sys.argv[2:])
        print(args)
        # Load Data
        train_set, test_set, val_set = dataset()
        classes = ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal', 'Trash']
        # Set Device and Model Configurations
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        model = ResNet()
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_dl = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_dl = DataLoader(val_set, args.batch_size * 2, shuffle=True, num_workers=4, pin_memory=True)
        # Initialize wandb
        wandb.login()
        wandb.init(project='testing', config=args, dir=str(project_dir.joinpath('./reports')))
        wandb.watch(model, log_freq=100)
        for epoch in range(args.num_epochs):
            # Training
            train_accuracy = 0
            model.train()
            for (counter, (images, labels)) in enumerate(train_dl):
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                _, predictions = torch.max(output, dim=1)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_acc = torch.tensor(torch.sum(predictions == labels).item() / len(predictions))
                wandb.log({"Training Accuracy (per batch)": train_acc})
                wandb.log({"Training Loss": loss})
                train_accuracy += train_acc
            epoch_train_acc = train_accuracy.item() / (counter + 1)
            wandb.log({"Training Accuracy (per epoch)": epoch_train_acc})
            print('(Epoch {}) Training accuracy: {:.2f}%'.format(epoch + 1, epoch_train_acc * 100))
            # Evaluation
            with torch.no_grad():
                val_accuracy = 0
                class_acc = {}
                correct_pred_list = [0 for _ in range(6)]
                total_pred_list = [0 for _ in range(6)]
                model.eval()
                for (counter, (images, labels)) in enumerate(val_dl):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predictions = torch.max(outputs, dim=1)
                    val_loss = criterion(outputs, labels)
                    val_acc = torch.tensor(torch.sum(predictions == labels).item() / len(predictions))
                    wandb.log({"Validation Accuracy (per batch)": val_acc})
                    wandb.log({"Validation Loss": val_loss})
                    val_accuracy += val_acc
                    for label, prediction in zip(labels, predictions):
                        if label == prediction:
                            correct_pred_list[label] += 1
                        total_pred_list[label] += 1
                epoch_val_acc = val_accuracy.item() / (counter + 1)
                wandb.log({"Validation Accuracy (per epoch)": epoch_val_acc})
                print('(Epoch {}) Validation accuracy: {:.2f}%'.format(epoch + 1, epoch_val_acc * 100))
            for i in range(len(classes)):
                print('Accuracy of %5s : %.2f %%' % (classes[i], 100 * correct_pred_list[i] / total_pred_list[i]))
                class_acc["Accuracy of %5s" % (classes[i])] = 100 * correct_pred_list[i] / total_pred_list[i]
            wandb.log(class_acc)
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
    sampler = optuna.samplers.TPESampler(seed=10)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10))
    study.optimize(objective, n_trials=5)
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image("Images/fig1.png")
    # fig.show()

    fig = optuna.visualization.plot_intermediate_values(study)
    fig.write_image("Images/fig2.png")
    # fig.show()
    # TrainOREvaluate()
