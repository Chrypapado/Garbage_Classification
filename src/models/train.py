# Import Libraries and Modules
import sys
from pathlib import Path
import wandb
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from omegaconf import OmegaConf
import hydra
from model import ResNet

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir / 'src/data'))
from make_dataset import main as dataset  # noqa: E402


@hydra.main(config_path="config", config_name="default_config.yaml")
def train(config):
    print("Training day and night")
    # Arguments
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.experiment
    # Load Data
    train_set, test_set, val_set = dataset()
    classes = ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal', 'Trash']
    # Set Device and Model Configurations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = ResNet()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])
    train_dl = DataLoader(train_set, hparams["batch_size"], shuffle=True, num_workers=hparams["num_workers"], pin_memory=True)
    val_dl = DataLoader(val_set, hparams["batch_size"] * 2, shuffle=True, num_workers=hparams["num_workers"], pin_memory=True)
    # Initialize wandb
    wandb.login()
    wandb.init(project='Garbage_Project', dir=str(project_dir.joinpath('./reports')))
    wandb.watch(model, log_freq=100)
    for epoch in range(hparams['num_epochs']):
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
            correct_pred_list = [0 for _ in range(len(classes))]
            total_pred_list = [0 for _ in range(len(classes))]
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


if __name__ == '__main__':
    train()
