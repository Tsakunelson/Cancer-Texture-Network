"""
PEP: 257
Title: Image Classifier Load Data class
Author: Nelson Zange Tsaku
Type: Informational
Description: This class provides transformed data into tensors, given the data directory path, in the form of dataLoaders with a given batch size.
Created: 07/18/2018
"""


from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np 
def load(directory):
    #locating data directories 
    data_dir = directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    #test_dir = data_dir + '/test_gt'
    
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomGrayscale(p=0.1),
                                          transforms.RandomRotation(30),
                                         transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.RandomResizedCrop(299),#224
                                          transforms.ToTensor(),
                                         transforms.Normalize([0.57806385, 0.57806385, 0.57806385],[0.00937135, 0.00937135, 0.00937135])])

    valid_transforms = transforms.Compose([
                                          transforms.Resize(300), #256
                                          transforms.RandomResizedCrop(299),#224
                                          transforms.ToTensor(),
                                         transforms.Normalize([0.57806385, 0.57806385, 0.57806385],[0.00937135, 0.00937135, 0.00937135])])
   # test_transforms = transforms.Compose([
   #                                       transforms.Resize(256),
   #                                       transforms.RandomResizedCrop(224),
   #                                       transforms.ToTensor(),
   #                                       transforms.Normalize([0.57806385, 0.57806385, 0.57806385],[0.00937135, 0.00937135, 0.00937135])
   ##                                       ])


    # TODO: Load the datasets with ImageFolder
    #print(train_dir)
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    
    #print(train_datasets)
    #print(train_datasets.__getitem__(10)[1])
    #test_datasets =  datasets.ImageFolder(test_dir, transform = test_transforms)
    
    #num_train = len(train_datasets)
    #indices = list(range(num_train))
    #split = int(np.floor(0.2*num_train))
    #np.random.seed(224)
    #np.random.shuffle(indices)
    #train_idx,valid_idx = indices[split:], indices[:split]
    #train_sampler = SubsetRandomSampler(train_idx)
    #valid_sampler = SubsetRandomSampler(valid_idx)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    #trainLoader = DataLoader(train_datasets, batch_size = 32,
    #                sampler=train_sampler, num_workers=4,
    #                pin_memory=False)
    #validLoader = DataLoader(train_datasets, batch_size = 32,
    #                sampler=valid_sampler, num_workers =4,
    #                pin_memory=False)
   # testLoader =  DataLoader(test_datasets, batch_size = 30, shuffle = False)
    
    #list_dataLoaders = [trainLoader, validLoader] # , testLoader]
    
    list_datasets = [train_datasets, valid_datasets] # , test_datasets]

    return list_datasets
#dataset = load("./poorly_cohessive/ck_dataset")
#print(dataset[0].__getitem__())

