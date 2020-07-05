import os
from torchvision import transforms
import torch
import glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import cfg
import time
# TODO: Define your data path (the directory containing the 4 np array files)
DATA_PATH = cfg['DATA_PATH']

class FMNIST(Dataset)
    def __init__(self, set_name,transform = None):
        super(FMNIST, self).__init__()
        # TODO: Retrieve all the images and the labels, and store them
        # as class variables. Maintaing any other class variables that 
        # you might need for the other class methods. Note that the 
        # methods depends on the set (train or test) and thus maintaining
        # that is essential.
        #Dataset folder structure -> Data -> trainimg.npy,trainlabel.npy, testlabel.npy,testimg.npy

        #torch.from_numpy(DATA_PATH+set_name+'_images.npy').float()

        #torch.from_numpy(DATA_PATH+set_name+'_labels.npy').float()
        self.labels = np.load(self.target)
        self.images = np.load(self.data)
        self.transform = transform

    def __len__(self):
        # TODO: Complete this
        return len(self.labels)
        # raise NotImplementedError
    
    def __getitem__(self, index):
        # TODO: Complete this'
        
        #x =  #query
        #y = 
        
        
        if self.transform:
            x = self.transform(x)
        return x, y
        raise NotImplementedError

def get_data_loader(set_name):
    # TODO: Create the dataset class tailored to the set (train or test)
    # provided as argument. Use it to create a dataloader. Use the appropriate
    # hyper-parameters from cfg
    transform = None
	# for train n test different transform
    if set_name is 'train':
    	transform = transforms.Compose([transforms.ToTensor()])
    	# pass
    	#add random horizontal flips etc
    else:
        
    	transform = transforms.Compose([transforms.ToTensor()])    	
    
    dataset = FMNIST(set_name,transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    #CPU
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #model.to(device)
    # raise NotImplementedError
    return loader