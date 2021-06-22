# Import Libraries and Modules
import sys
import joblib
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from azureml.core import Run

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir / 'src/data'))
from make_dataset import main as dataset  # noqa: E402

sys.path.insert(0, str(project_dir / 'src/models'))
from model import ResNet  # noqa: E402

# Get the experiment run context
run = Run.get_context()

# Arguments
parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--lr', default=5.5e-5)
parser.add_argument('--batch_size', default=32)
parser.add_argument('--num_epochs', type=int, default=10)
args = parser.parse_args(sys.argv[2:])

# Load Data
train_set, test_set, val_set = dataset()
classes = ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal', 'Trash']

# Set Device and Model Configurations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
train_dl = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_set, args.batch_size * 2, shuffle=True, num_workers=4, pin_memory=True)

# Training
for epoch in range(args.num_epochs):
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
        run.log('Training Accuracy (per batch)', np.float(train_acc))
        run.log('Training Loss (per batch)', np.float(loss))
        train_accuracy += train_acc
    epoch_train_acc = train_accuracy.item() / (counter + 1)
    run.log('Training Accuracy', np.float(epoch_train_acc))
    # Evaluation
    with torch.no_grad():
        val_accuracy = 0
        class_acc = [0 for _ in range(len(classes))]
        correct_pred_list = [0 for _ in range(len(classes))]
        total_pred_list = [0 for _ in range(len(classes))]
        model.eval()
        for (counter, (images, labels)) in enumerate(val_dl):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, dim=1)
            val_loss = criterion(outputs, labels)
            val_acc = torch.tensor(torch.sum(predictions == labels).item() / len(predictions))
            run.log('Validation Accuracy (per batch)', np.float(val_acc))
            run.log('Validation Loss (per batch)', np.float(val_loss))
            val_accuracy += val_acc
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred_list[label] += 1
                total_pred_list[label] += 1
        epoch_val_acc = val_accuracy.item() / (counter + 1)
        run.log('Validation Accuracy', np.float(epoch_val_acc))
    for i in range(len(classes)):
        class_acc[i] = 100 * correct_pred_list[i] / total_pred_list[i]
        run.log(classes[i], np.float(class_acc[i]))
    joblib.dump(value=model.state_dict(), filename=project_dir.joinpath('models/azure/model' + str(epoch) + '.pth'))

run.complete()
