"""
PEP: 257
Title: Image Classifier helper class
Author: Nelson Zange Tsaku
Type: Informational
Description: This class provides useful functions such as new_model() to create the model, get_model(), to load a checkpoint for a model and process_image() to process the input image for prediction.
Created: 07/18/2018
"""
import torch
from torchvision import models
from collections import OrderedDict
from torchvision import models
from collections import OrderedDict
from torch import nn
import numpy as np
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self,architecture, num_classes=2):
        super().__init__()
        #self.conv1 = nn.Conv2d(3, 6, kernel_size=11)
        #nn.BatchNorm2d(32)
        #self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        #self.conv2 = nn.Conv2d(6, 12, kernel_size=11)
        #nn.BatchNorm2d(128)
        #self.fc1 = nn.Linear(12*47*47, 120)
        #self.fc2 = nn.Linear(120, 60)
        #self.fc3 = nn.Linear(60, 2)
        #if architecture == 'inceptionV3':
        self.model = models.inception_v3(pretrained = False)
        #num_ftrs = self.model.fc.in_features
        #self.pool1 = nn.GlobalAveragePooling2D()
        #self.pool = nn.AdaptiveAvgPool2d((5,7))
        self.fc = nn.Linear(224, 2*3*224)
        self.fc1 = nn.Linear(2*3*224, 60)
        self.fc2 = nn.Linear(60, 2)
           #self.model.fc = nn.Linear(512,2)
        
        
    def forward(self, X):
        #out = self.pool(X)
        #out = self.pool(F.relu(self.conv2(out)))
        print(len(X))
        print(X.shape)
        #print(X.shape)
        #print(X[0].shape)
        #X = Variable(X.data, requires_grad=True)
        #X = np.stack(X).toTensor
        
        #out = X.view(X.size(0), -1)
        #print(X.shape)
        #print(out.shape)
        
        out = F.relu(self.fc(X))
        out = F.relu(self.fc1(out))
        out = F.softmax(self.fc2(out)) 
        return out  #self.model_ft(X) #out

def new_model(architecture, num_hidden_layers ):
    
    # TODO: Build and train your network
    #train with vgg13 model if vgg13 is passed as parameter
    #if architecture == 'vgg13':
    #     model = models.vgg13(pretrained = True)
    
    #train with vgg16 model if vgg16 is passed as parameter
    #if architecture == 'vgg16':
    #    model = models.vgg16(pretrained = True)
        
    #if architecture == 'inceptionV3':
    #    model = models.inception_v3(pretrained = False)
    
    #if architecture == 'U-net':
    #    model = models.u-noet(pretrained=True)

    #define classifier without pretrained model
    input_size = 3*224*224
    hidden_size = [512, 128, 64]
    output_size = 2
    
    model = ConvNet(architecture, output_size)
    #classifier = nn.Sequential(nn.Linear(input_size, hidden_size[0]),
    #        nn.ReLU(),
    #        nn.Linear(hidden_size[0],hidden_size[1]),
    #        nn.ReLU(),
    #        nn.Linear(hidden_size[1], hidden_size[2]),
    #        nn.ReLU(),
    #        nn.Linear(hidden_size[2], output_size),
    #        nn.Softmax(dim=1))
    #classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 512 )), 
    #                             ('relu', nn.ReLU()), 
    #       ('fc2', nn.Linear(512 , output_size)),
    #                  ('output', nn.LogSoftmax(dim=1))
    #                      ]))
    #model.classifier = classifier
    
            
    
    #classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, num_hidden_layers )), 
     #                            ('relu', nn.ReLU()), 
      #               ('dropout', nn.Dropout(p=0.4)),  #was 0.337
       #    ('fc2', nn.Linear(num_hidden_layers , 2)),
        #              ('output', nn.LogSoftmax(dim=1))
         #                 ]))

    #bind classifier to model
    #model.classifier = classifier
  
    return model

def get_model(path):
    last_checkpoint = torch.load(path)
    
    model  = new_model(last_checkpoint['arch'], last_checkpoint['num_hidden_layers'])
    
    #restore model from states
    model.state_dict = last_checkpoint['model_state'] 
    model.criterion = last_checkpoint['criterion_state']
    model.optimizer_state = last_checkpoint['optimizer_state']
    model.class_to_idx = last_checkpoint['class_to_idx']
    model.epochs = last_checkpoint['epochs']
    model.Best_train_loss = last_checkpoint['Best_train_loss']
    model.Best_train_accuracy = last_checkpoint['Best_train_accuracy']
    model.Best_Validation_loss = last_checkpoint['Best_Validation_loss']
    model.Best_Validation_accuracy = last_checkpoint['Best_Validation_accuracy']
    
    return model
    
#process image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    size = 50,50
    image.thumbnail(size)
    
    width, height = image.size
    
    new_width, new_height = 224, 224   # Get dimensions
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    temp_image = image.crop((left, top, right, bottom))
    image_array = np.array(temp_image)
    
    means = [0.57806385, 0.57806385, 0.57806385]
    std = [0.00937135, 0.00937135, 0.00937135]
    
    image_array = (image_array - means)/std
    
    image_array = np.transpose(image_array,(2,1,0))
    
    torch_tensor = torch.from_numpy(image_array)
    torch_tensor = torch.unsqueeze(torch_tensor, 0)
    
    return torch_tensor.float()
    
    
    
    

    
   
