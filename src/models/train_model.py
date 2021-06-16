import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torchvision import datasets,transforms

from pathlib import Path
from model import MyAwesomeModel

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
        parser.add_argument('--lr', default=0.1)
        #Add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        #Write Files
        filepath = str(project_dir.joinpath('reports'))
        fpTL_batch = open(filepath + "/Train_Loss_Batch" , "w")
        fpTL_epoch = open(filepath + "/Train_Loss_Epoch" , "w")
        #Training loop
        #device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #print(device)
        model = MyAwesomeModel()
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_set, test_set, val_set = dataset()
        epochs = 5
        #steps = 0
        train_losses = []
        epoch_losses = []
        for epoch in range(epochs):
            print("At epoch: ", epoch)
            epoch_loss = 0
            #num_batches = train_set.__len__()
            #running_loss = 0
            for (i, (images, labels)) in enumerate(train_set):
                #images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                #steps += 1
                train_losses.append(loss.item()/64)
                epoch_loss += loss.item()
                #print('\rProcessing batch %04d out of %04d, running loss %04f' % (i, num_batches, loss))
            print(f"Training loss: {epoch_loss/len(train_set)}")
            epoch_losses += [epoch_loss]
            epoch_loss = 0
            #torch.save(model.state_dict(), project_dir.joinpath('models/model.pth' + '_' + str(epoch+1)))
        fpTL_batch.write(str(train_losses) + ',')
        fpTL_batch.flush()
        fpTL_epoch.write(str(epoch_losses) + ',')
        fpTL_epoch.flush()
        torch.save(model.state_dict(), project_dir.joinpath('models/model.pth'))

        # Plot resulting training losses
        #fig, ax = plt.subplots(1, 2, figsize=(10,4))
        #ax[0].plot(np.arange(1, steps+1), train_losses, label='Training Losses (per batch)')
        #ax[1].plot(np.arange(1, len(epoch_losses)+1), epoch_losses, label='Training Losses (per epoch)', color="#F58A00")
        #ax[0].legend()
        #ax[1].legend()
        #plt.tight_layout()
        #image_path = str(project_dir / 'reports/figures/Training_loss.png')
        #plt.savefig(image_path)
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        dict_ = torch.load(project_dir.joinpath('models/model.pth'))
        model.load_state_dict(dict_)
        train_set, test_set, val_set = dataset()
        accuracy = 0
        counter = 0
        # turn off gradients for the purpose of speeding up the code
        with torch.no_grad():
            for images, labels in test_set: # with batch size 64
                ps = torch.exp(model(images))
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                counter += 1
                accuracy += torch.mean(equals.type(torch.FloatTensor))
            accuracy = accuracy / counter
            print(f'Accuracy: {accuracy.item()*100}%')

if __name__ == '__main__':
    TrainOREvaluate()