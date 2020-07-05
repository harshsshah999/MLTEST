from torchvision import transforms
from datetime import datetime
                    
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


from data import get_data_loader
from network import Network,xavier_init
from config import cfg

try:
    from termcolor import cprint
except ImportError:
    cprint = None

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def get_lr(optimizer):
    #TODO: Returns the current Learning Rate being used by
    # the optimizer
    for param_group in optimizer.param_groups:
        return param_group['lr']

class AvgMeter():
    def __init__(self):
        self.qty = 0
        self.cnt = 0
    
    def update(self, increment, count):
        self.qty += increment
        self.cnt += count
    
    def get_avg(self):
        if self.cnt == 0:
            return 0
        else: 
            return self.qty/self.cnt


def run(net, epoch, loader, optimizer, criterion, scheduler, train=True):
    # TODO: Performs a pass over data in the provided loader
    
    avgloss = AvgMeter()
    
    if train is False:
        accuracy = AvgMeter()
    for idx,(images, labels) in enumerate(loader):
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1).float()
        images,labels = images.cuda(),labels.cuda()

        # Training pass

        if train is True:
            net.train()
            optimizer.zero_grad()
        else:
            net.eval()
        
        output = net(images)
        loss = criterion(output, labels.long())
        if train is True:
            loss.backward()
            optimizer.step()
        if train is False:
            acc = (labels.long() == torch.argmax(output,dim=1)).float().sum()
            accuracy.update(acc,len(labels))
        avgloss.update(loss.item(),1) #loss per batch
    if train is False:
        return avgloss.get_avg(),accuracy.get_avg() #average loss per image
    else:
        return avgloss.get_avg(),None
    # TODO: Initalize the different Avg Meters for tracking loss and accuracy (if test)#done

    # TODO: Iterate over the loader and find the loss. Calculate the loss and based on which
    # set is being provided update you model. Also keep track of the accuracy if we are running
    # on the test set.#done

    # TODO: Log the training/testing loss using tensorboard.#done 
    
    # TODO: return the average loss, and the accuracy (if test set) #done


def train(net, train_loader, test_loader):    
    optimizer = optim.SGD(net.parameters(), lr= cfg['lr'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'], nesterov=cfg['nesterov'])
    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience = cfg['patience'],factor = cfg['lr_decay'])
    net=net.cuda()

    for i in range(cfg['epochs']):
        # Run the network on the entire train dataset. Return the average train loss
        # Note that we don't have to calculate the accuracy on the train set.
        
        loss,__ = run(net, i, train_loader, optimizer, criterion, scheduler)
        current_lr = get_lr(optimizer)
        print("epoch,lr and loss are",i+1,current_lr,loss)
        # Evaluate our model and add visualizations on tensorboard
        if i % cfg['val_every'] == 0:
        # Run the network on the test set, and get the loss and accuracy on the test set 
            testloss, acc = run(net, i, test_loader, optimizer, criterion, scheduler, train=False)
            print("Epoch: %d, Test Accuracy:%2f , test loss: %f" % (i+1, acc*100.0,testloss))



if __name__ == '__main__':
    # TODO: Create a network object
    net = Network()#normal
    # net= Network('xavier') 
    # net = Network('zero')#zero init
    #net = Network('one')

    # TODO: Create a tensorboard object for logging
    # TODO: Create train data loader
    train_loader = get_data_loader('train')

    # TODO: Create test data loader
    test_loader = get_data_loader('test')

    # Run the training!
    train(net, train_loader, test_loader)