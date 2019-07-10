"""
PEP: 257
Title: Image Classifier Predict class
Author: Nelson Zange Tsaku
Type: Informational
Description: This class performs predictions, given the apropriate hyperparameters (else the defaults are used), then returns the top K prediction probabilities and with the corresponding classes
Created: 07/18/2018
"""

import helper
from PIL import Image
import numpy as np
import argparse
import torch 
import json
import os
import pandas as pd
import numpy as np
from skorch import NeuralNetClassifier
from gridSearchSkorch4 import Inception3
import torch.nn as nn
import torch.optim as optim
from torchvision import models

######################################################################
#parse arguments
parser = argparse.ArgumentParser(description='Collecting Prediction parameters.')
parser.add_argument('--image_dir', type = str, default = './well_diff/HE/test_gt/', help = 'path to the dataset. Default = flowers/test/1/image_06743.jpg')
parser.add_argument('--checkpoint', type = str,default = './wd_heresults/0best_model.pt', help = 'Load checkpoint. Default is at checkpoint.pth')
parser.add_argument('--topk', type = int, default = 2, help = 'Top most prediction probabilities. Default = 3')
parser.add_argument('--labels', type = str, default = 'cat_to_name.json', help = 'Class to index labels. Default = cat_to_name.json')
parser.add_argument('--gpu', action ='store_true', default = False, help = 'Enable GPU mode. Default = True')
#########################################################################

args = parser.parse_args()

def predict(image_path, checkpoint, topk, labels, gpu):
    ''' Predict the class (or classes) of an image using aa trained deep learning model.
    '''
    #model = helper.get_model(checkpoint)
    #new_net = NeuralNetClassifier(
    #    module=models.Inception_v3,
    #    criterion=torch.nn.CrossEntropyLoss,
    #    optimizer= optim.SGD,
    #    )
    #new_net.initialize() 
    
    #new_net.load_params(f_params= checkpoint)
    model = models.Inception_v3.load_state_dict(torch.load(checkpoint)['valid_acc_best'])
    print (model.eval())
    #torch.load(path)

    #if gpu:
    #    model = model.cuda()
    #    torch.cuda.set_device(3)
    #loadvalid_acc_best
    image = Image.open(image_path)
    #process Image into a tensor
    image_tensor = helper.process_image(image)
    #if torch.cuda.is_available() and gpu:
    #    image = image_tensor.to('cuda')
    
    with torch.no_grad():
        output = model.forward(image)
    #else:
    output = model.forward(image_tensor)
    #get the probability output 
    probs = torch.exp(output)
    #print(np.isnan(probs.data.numpy()).sum())
    top_probs, indices = probs.topk(topk, sorted = True)
    print("Probabilities: " + str(top_probs))
    print("Indices: " + str(indices)) 
    
    if torch.cuda.is_available() and gpu:
        # Added softmax here as per described here:
        # https://github.com/pytorch/vision/issues/432#issuecomment-368330817
        probs = torch.nn.functional.softmax(top_probs.data, dim=1).cpu().numpy()[0]
        classes = indices.data.cpu().numpy()[0]
    else:       
        probs = torch.nn.functional.softmax(top_probs.data, dim=1).numpy()[0]
        classes = indices.data.numpy()[0]
    
    print(classes)    
    #print(probs)
    
    print(probs)
    
    return probs, classes

# Make predictions if called from command linesvs = []
cancer_files = []
normal_files = []

for filename in os.listdir("./well_diff/HE/test_gt/0/"):
    #if filename.startswith("2_0"):
    cancer_files.append(filename)
    
    #if filename.startswith("1_1"):
     #   cancer_roi1.append(filename)
for filename in os.listdir("./well_diff/HE/test_gt/1/"):
    normal_files.append(filename)       
#total_samples = len(cancer_roi0)+len(noncancer)


###########################Cancer regions#######################################################################
k = 0
cancer_df = pd.DataFrame(columns=["Name","Actual","Predicted","Probability"])
for temp in cancer_files:
    #print("Noncancer")
    probs, classes = predict(args.image_dir+str('0/')+temp, args.checkpoint, args.topk, args.labels,args.gpu)
    cancer_df.loc[k] = [temp,"noncancer",classes[0], probs[0]]
    k += 1
#print("Predictions for {} with classes {}: has Probability {} Respectively".format(args.topk,classes,probs))
cancer_df.to_csv("wd_cancer_HEresults.csv")



###########################Normal regions##########################################################################
normal_df = pd.DataFrame(columns=["Name","Actual","Predicted","Probability"])
i = 0
for temp in normal_files:
    probs, classes = predict(args.image_dir+str('1/')+temp, args.checkpoint, args.topk, args.labels,args.gpu)
    normal_df.loc[i] = [temp,"cancer",classes[0], probs[0]]
    i += 1
normal_df.to_csv(" wd_normal_HEresults.csv")






  
