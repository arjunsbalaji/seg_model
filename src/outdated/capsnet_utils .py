#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 15:07:19 2018

@author: arjunbalaji
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 16:45:12 2018

@author: arjun
"""

#caps net idea for aortic dissection thrombus profiler 
#not worked on yet

import numpy as np
import os
import torch 
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable
import time 
import matplotlib.pyplot as plt
import shutil
from sklearn import preprocessing
from matplotlib import pyplot

start_time = time.time()

os.chdir('..')
print(os.getcwd())
#main_data_dir = os.path.join(os.getcwd(), 'data')

#get the data paths
#FL_data_path = os.path.join(main_data_dir, 'FL')
#ILT_data_path = os.path.join(main_data_dir, 'ILT')
#Image_data_path = os.path.join(main_data_dir, 'Images')
#TL_data_path = os.path.join(main_data_dir, 'TL')

def get_image(main_data_dir, name, image_type):
    this_data_path = os.path.join(main_data_dir, image_type)
    return np.load(os.path.join(this_data_path, name))

###############################################################################
#need this special squash function for caps net

def squash(s, axis = -1, epsilon = 1e-7):
    squared_norm = torch.sum(s*s, dim = axis)

    safe_norm = torch.sqrt(squared_norm + epsilon)

    squash_factor = squared_norm / (1. + squared_norm)
    
    #the unsqueeze here is very important!
    #for safe norm to be broadcasted appropriately 
    unit_vector = torch.div(s, safe_norm.unsqueeze(-1))
    
    #for squash factor to be broadcasted appropriately 
    return torch.mul(squash_factor.unsqueeze(-1), unit_vector)

###############################################################################
#dataset class
class AorticDissectionDataset(Dataset):
    """
    First we create a dataset that will encapsulate our data. It has 3 special 
    functions which will be explained as they go. We will pass this dataset object
    to the torch dataloader object later which will make training easier.
    """
    def __init__ (self,
                  main_data_dir,
                  transform = None):
        self.main_data_dir = main_data_dir
        self.transform = transform
        
        #iterate through the 2d images and get all their names
        name_list = []
        for im in os.listdir(os.path.join(self.main_data_dir, 'FL')):
            filename = os.fsdecode(im)
            name_list.append(filename)
            
        self.name_list = name_list
        
        
    def __getitem__(self, idx):
        """This function will allow us to index the data object and it will 
        return a sample."""
        name = self.name_list[idx]
        
        #label
        #TL = get_image(name, 'TL')
        #FL = get_image(name, 'FL')
        #ILT = get_image(name, 'ILT')      
        label = np.array([get_image(self.main_data_dir, name, 'TL'),
                          get_image(self.main_data_dir, name, 'FL'),
                          get_image(self.main_data_dir, name, 'ILT')])
        #image data and filters

        Image = get_image(self.main_data_dir, name, 'Images')

        Image = preprocessing.scale(Image)
        #og = preprocessing.MinMaxScaler(og)
        
        sample = {'input': torch.tensor(Image),
                  'label': torch.tensor(label),
                  'case_name': name}
        """
        #create a sample as a dictionary
        sample = {'og': torch.tensor(og),
                  'double': torch.tensor(double),
                  'single': torch.tensor(single),
                  'longgrad': torch.tensor(longgrad),
                  'objective': torch.tensor(obj),
                  'case_name': name}
        """  
        return sample
    
    def __len__(self):    
        """This function is mandated by Pytorch and allows us to see how many 
        data points we have in our dataset"""
        return len(self.name_list)

###############################################################################
class Get_Primary_Caps(torch.nn.Module):
    """This is the primary caps block. It takes in an input image of 1 channel
    and 512x512 pixels and outputs a caps1_n_caps primary caps each which is a
    caps1_n_dims dimensional vector. there is work to be done here so that 
    the numbers all make themselves work. at the moment you have to carefully 
    check each one to make sure the model runs"""
    def __init__(self, caps1_n_maps,
                 caps1_caps_grid_ydim,
                 caps1_caps_grid_xdim,
                 caps1_n_dims):
        super(Get_Primary_Caps, self).__init__()
        self.caps1_n_maps = caps1_n_maps 
        self.caps1_caps_grid_ydim = caps1_caps_grid_ydim
        self.caps1_caps_grid_xdim = caps1_caps_grid_xdim
        self.caps1_n_dims = caps1_n_dims 
        
        self.relu = torch.nn.ReLU()
        
        conv1_parameters = {'i': 1, 'o': 32, 'k': 2, 's': 2}
        self.conv1 = torch.nn.Conv2d(in_channels=conv1_parameters['i'],
                                     out_channels=conv1_parameters['o'],
                                     kernel_size=conv1_parameters['k'],
                                     stride=conv1_parameters['s'])
        
        
        
        conv2_parameters = {'i': 32, 'o': self.caps1_n_dims * self.caps1_n_maps, 'k': 2, 's': 2}
        self.conv2 = torch.nn.Conv2d(in_channels=conv2_parameters['i'],
                                     out_channels=conv2_parameters['o'],
                                     kernel_size=conv2_parameters['k'],
                                     stride=conv2_parameters['s'])
        
        
        
    def forward(self, x):
        x = self.conv1(x)
        #print(x.size())
        x = self.relu(x)
        x = self.conv2(x)
        #print(x.size())
        x = self.relu(x)
        x = x.view([-1,
                    self.caps1_n_maps,
                    self.caps1_caps_grid_ydim, 
                    self.caps1_caps_grid_xdim, 
                    self.caps1_n_dims])
        x = squash(x)
        return x

###############################################################################
class Get_Abstract_Caps_Down(torch.nn.Module):
    """This is the abstract caps layer. We take in an input of the capsules
    of the previous layer and then output predictions of abstract capsules."""
    def __init__(self, 
                 batch_size,
                 capsin_n_maps,
                 capsin_n_dims,
                 capsout_n_maps,
                 capsout_n_dims,
                 old_h,
                 old_w,
                 y_kernel,
                 x_kernel):
        super(Get_Abstract_Caps_Down, self).__init__()
        
        self.batch_size = batch_size
        self.capsin_n_dims = capsin_n_dims
        self.capsin_n_maps = capsin_n_maps 
        self.capsout_n_maps = capsout_n_maps 
        self.capsout_n_dims = capsout_n_dims
        
        self.old_h = old_h
        self.old_w = old_w
        self.y_kernel = y_kernel
        self.x_kernel = x_kernel
        
        self.W_init = torch.nn.Parameter(torch.Tensor(self.batch_size,
                                                      self.capsin_n_maps,
                                                      1,
                                                      self.y_kernel,
                                                      self.x_kernel,
                                                      self.capsout_n_maps,
                                                      self.capsout_n_dims,
                                                      self.capsin_n_dims))
        
        self.reset_parameters()
        
        self.predict_vectors = Predict_Vectors_Down(self.W_init)
        
        self.routing = Agreement_Routing_Down(input_caps_maps = self.capsin_n_maps,
                                         input_caps_dim = self.capsin_n_dims,
                                         output_caps_maps = self.capsout_n_maps,
                                         output_caps_dim = self.capsout_n_dims,
                                         new_hl = self.old_h / self.y_kernel,
                                         new_wl = self.old_w / self.x_kernel,
                                         num_iterations = 2)


    
    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.capsin_n_maps)
        self.W_init.data.uniform_(-stdv, stdv)

    def forward(self, primary_caps):
        
        x = self.predict_vectors(primary_caps)
        #print(x.size())
        x = self.routing(x)
        
        return x
###############################################################################

class Predict_Vectors_Down(torch.nn.Module):
    """This module takes in a 5 dimensional (1 batch + 4 dim) input tensor that
    represents the capsule vectors of the higher layer and applies a six 
    dimensional weight tensor on it in a kernel fashion resulting in a set of 
    prediction vectors per capsule vector in the new layer"""
    def __init__(self, weights):
        super(Predict_Vectors_Down, self).__init__()
        self.weights = weights
        
        
    def forward(self, input_tensor):
        batch_sizeit, input_mapsit, hl, wl, input_caps_dimit  = input_tensor.size()
        
        batch_sizew, input_mapsw, _, y_kernel, x_kernel, output_maps, output_caps_dim, input_caps_dimw= self.weights.size()
        
        
        if input_caps_dimit != input_caps_dimw:
            print('Input_tensor and Weights dont have the same input capsule dim')
        elif batch_sizeit != batch_sizew:
            print('Input_tensor and Weights dont have the same batch size')
        elif input_mapsit != input_mapsw:
            print('Input tensor and weights dont have the same no. of maps')
            
        
        #print('hl=', hl, 'wl=',wl)
        
        #output_pred_vectors = torch.empty(0)
        
        #print(input_tensor.size(), '\t', 'input tensor size')
        #print( self.weights.size(), '\t', 'weights size')
     
        input_tensor = input_tensor.view([-1,
                                          input_mapsit,
                                          int((hl / y_kernel) * (wl / x_kernel)),
                                          y_kernel,
                                          x_kernel,
                                          1,
                                          input_caps_dimit,
                                          1])
    
        input_tensor = torch.matmul(self.weights, input_tensor)
        
        
        #print(input_tensor.size(), 'final predictions size')
        #could probably make these for loops more efficient by rolling them into one 
        input_tensor = input_tensor.sum(dim=3)
        input_tensor = input_tensor.sum(dim=3)
        
        
        input_tensor = input_tensor.view([-1,
                                          output_maps,
                                          int(hl / y_kernel),
                                          int(wl / x_kernel),
                                          input_mapsit,
                                          output_caps_dim])
        
    
        #print(input_tensor.size(), 'final predictions size')

        return input_tensor


###############################################################################

class Agreement_Routing_Down(torch.nn.Module):
    """This is the localised agreement routing algorithm. It takes in the total
    prediction vectors from a layer l and computes the routing weights for 
    those predictions. It then squashes the prediction vectors using the 
    custom squash function."""
    def __init__(self, input_caps_maps,
                 input_caps_dim,
                 output_caps_maps,
                 output_caps_dim,
                 new_hl,
                 new_wl,
                 num_iterations):
        super(Agreement_Routing_Down, self).__init__()
        self.input_caps_maps = input_caps_maps
        self.input_caps_dim = input_caps_dim
        self.output_caps_maps = output_caps_maps
        self.output_caps_dim = output_caps_dim 
        self.new_hl = int(new_hl)
        self.new_wl = int(new_wl)
        self.num_iterations = num_iterations 
        self.softmax = torch.nn.Softmax(dim = -1)
        
        
        self.b = torch.nn.Parameter(torch.zeros(1,
                                                self.output_caps_maps,
                                                self.new_hl,
                                                self.new_wl,
                                                self.input_caps_maps))
        
    def forward(self, tensor_of_prediction_vector):
        
        c = self.softmax(self.b)

        #print(c.size(),'c')
        #print(tensor_of_prediction_vector.size(), 'pred vectors')
        
        output_vectors = torch.mul(tensor_of_prediction_vector,
                                                    c.unsqueeze(-1))

        
        
        output_vectors = output_vectors.sum(dim=-2)
        
        
        #print(output_vectors.size())
        output_vectors = squash(output_vectors, axis = -1)
        
        b_batch = self.b
        
        for d in range(self.num_iterations):
            #print('meme')
            #print('preds', tensor_of_prediction_vector.size())
            #print('out', output_vectors.size())
            
            b_batch = b_batch + torch.mul(tensor_of_prediction_vector,
                                output_vectors.unsqueeze(-2)).sum(dim = -1)
            
            #b_batch = b_batch.sum(dim = -1)
            
            #b_batch = 
            #print('bbatch', b_batch.size())
            
            c = self.softmax(b_batch)
            
            output_vectors = torch.mul(tensor_of_prediction_vector,
                                       c.unsqueeze(-1))
            
            output_vectors = output_vectors.sum(-2)
            output_vectors = squash(output_vectors,
                                                 axis = -1)
            #print(c.size())
            
        #print(tensor_of_prediction_vector.size())
        #print(output_vectors.size())
        return output_vectors
    
###############################################################################
class Get_Abstract_Caps_Up(torch.nn.Module):
    """This is the abstract caps layer. We take in an input of the capsules
    of the previous layer and then output predictions of abstract capsules."""
    def __init__(self,
                 batch_size,
                 capsin_n_maps,
                 capsin_n_dims,
                 capsout_n_maps,
                 capsout_n_dims,
                 old_h,
                 old_w,
                 y_kernel,
                 x_kernel):
        super(Get_Abstract_Caps_Up, self).__init__()
        
        
        self.batch_size = batch_size
        self.capsin_n_dims = capsin_n_dims
        self.capsin_n_maps = capsin_n_maps 
        self.capsout_n_maps = capsout_n_maps# 32x32 grid of vectors are abstract caps
        self.capsout_n_dims = capsout_n_dims
        
        self.old_h = old_h
        self.old_w = old_w
        self.y_kernel = y_kernel
        self.x_kernel = x_kernel
        
        self.W_init = torch.nn.Parameter(torch.Tensor(self.batch_size,
                                                      self.capsin_n_maps,
                                                      1,
                                                      self.y_kernel,
                                                      self.x_kernel,
                                                      self.capsout_n_maps,
                                                      self.capsout_n_dims,
                                                      self.capsin_n_dims))
        
        self.reset_parameters()
        self.predict_vectors = Predict_Vectors_Up(self.W_init,
                                               [self.y_kernel,self.x_kernel])
        
        self.routing = Agreement_Routing_Up(input_caps_maps = self.capsin_n_maps,
                                         input_caps_dim = self.capsin_n_dims,
                                         output_caps_maps = self.capsout_n_maps,
                                         output_caps_dim = self.capsout_n_dims,
                                         new_hl = self.old_h * self.y_kernel,
                                         new_wl = self.old_w * self.x_kernel,
                                         num_iterations = 2)


    
    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.capsin_n_maps)
        self.W_init.data.uniform_(-stdv, stdv)

    def forward(self, primary_caps):
        
        x = self.predict_vectors(primary_caps)
        x = self.routing(x)
        
        return x
###############################################################################

class Predict_Vectors_Up(torch.nn.Module):
    """This module takes in a 5 dimensional (1 batch + 4 dim) input tensor that
    represents the capsule vectors of the higher layer and applies a six 
    dimensional weight tensor on it in a kernel fashion resulting in a set of 
    prediction vectors per capsule vector in the new layer"""
    def __init__(self, weights, kernels):
        super(Predict_Vectors_Up, self).__init__()
        self.weights = weights
        self.kernels = kernels
        
        
    def forward(self, input_tensor):
        batch_sizeit, input_mapsit, hl, wl, input_caps_dimit  = input_tensor.size()
        
        batch_sizew, input_mapsw, _, y_kernel, x_kernel, output_maps, output_caps_dim, input_caps_dimw= self.weights.size()
        
        
        if input_caps_dimit != input_caps_dimw:
            print('Input_tensor and Weights dont have the same input capsule dim')
        elif batch_sizeit != batch_sizew:
            print('Input_tensor and Weights dont have the same batch size')
        elif input_mapsit != input_mapsw:
            print('Input tensor and weights dont have the same no. of maps')
            
        
        #print('hl=', hl, 'wl=',wl)
        
        #output_pred_vectors = torch.empty(0)
        
        #print(input_tensor.size(), '\t', 'input tensor size')
        #print( self.weights.size(), '\t', 'weights size')
        
        input_tensor = input_tensor.view([-1,
                                          input_mapsit,
                                          int((hl / y_kernel) * (wl / x_kernel)),
                                          1,
                                          1,
                                          1,
                                          input_caps_dimit,
                                          1])
    
        input_tensor = torch.matmul(self.weights, input_tensor)
        
        
        #print(input_tensor.size(), 'final predictions size')
        #could probably make these for loops more efficient by rolling them into one 

        input_tensor = input_tensor.view([-1,
                                          output_maps,
                                          int(hl * y_kernel),
                                          int(wl * x_kernel),
                                          input_mapsit,
                                          output_caps_dim])
        
    
        #print(input_tensor.size(), 'final predictions size')

        return input_tensor

###############################################################################

class Agreement_Routing_Up(torch.nn.Module):
    """This is the localised agreement routing algorithm. It takes in the total
    prediction vectors from a layer l and computes the routing weights for 
    those predictions. It then squashes the prediction vectors using the 
    custom squash function."""
    def __init__(self, input_caps_maps,
                 input_caps_dim,
                 output_caps_maps,
                 output_caps_dim,
                 new_hl,
                 new_wl,
                 num_iterations):
        super(Agreement_Routing_Up, self).__init__()
        self.input_caps_maps = input_caps_maps
        self.input_caps_dim = input_caps_dim
        self.output_caps_maps = output_caps_maps
        self.output_caps_dim = output_caps_dim 
        self.new_hl = int(new_hl)
        self.new_wl = int(new_wl)
        self.num_iterations = num_iterations 
        self.softmax = torch.nn.Softmax(dim = -1)
        
        self.b = torch.nn.Parameter(torch.zeros(1,
                                                self.output_caps_maps,
                                                self.new_hl,
                                                self.new_wl,
                                                self.input_caps_maps))
        
    def forward(self, tensor_of_prediction_vector):
        
        c = self.softmax(self.b)
        
        #print('c', c.size())
        #print(tensor_of_prediction_vector.size())
        output_vectors = torch.mul(tensor_of_prediction_vector,
                                                    c.unsqueeze(-1))
        #print('c', c.size())
        #print(tensor_of_prediction_vector.size())
        
        output_vectors = output_vectors.sum(dim=-2)
        
        
        #print(output_vectors.size())
        output_vectors = squash(output_vectors, axis = -1)
        
        b_batch = self.b
        
        for d in range(self.num_iterations):
            #print('meme')
            #print('preds', tensor_of_prediction_vector.size())
            #print('out', output_vectors.size())
            
            b_batch = b_batch + torch.mul(tensor_of_prediction_vector,
                                output_vectors.unsqueeze(-2)).sum(dim = -1)
            
            #b_batch = b_batch.sum(dim = -1)
            
            #b_batch = 
            #print('bbatch', b_batch.size())
            
            c = self.softmax(b_batch)
            
            output_vectors = torch.mul(tensor_of_prediction_vector,
                                       c.unsqueeze(-1))
            
            output_vectors = output_vectors.sum(-2)
            output_vectors = squash(output_vectors,
                                                 axis = -1)
            #print(c.size())
            
        #print(tensor_of_prediction_vector.size())
        #print(output_vectors.size())
        return output_vectors


###############################################################################    
    
class CapsNet(torch.nn.Module):
    """This is the actual model. It has a down line, a bottom pass and then a 
    series of up passes. On the up passes we concatenate prior down passes as 
    this improves the networks localisation. it is important the the tensors we 
    concatenate are the same size. so we use upsampling. Also be aware of the 
    channels, we want a lot of channels (~1000) so the network learns intricate
    features."""
    def __init__(self, batch_size):
        super(CapsNet, self).__init__()
        self.batch_size = batch_size
        
        self.get_prim_caps = Get_Primary_Caps(caps1_n_maps = 8,
                                              caps1_caps_grid_ydim = 128,
                                              caps1_caps_grid_xdim = 128,
                                              caps1_n_dims = 3)
        
        self.get_abstract_caps1 = Get_Abstract_Caps_Down(self.batch_size,
                                                         capsin_n_maps = 8,
                                                         capsin_n_dims = 3,
                                                         capsout_n_maps = 10,
                                                         capsout_n_dims = 5,
                                                         old_h = 128,
                                                         old_w = 128,
                                                         y_kernel = 2,
                                                         x_kernel = 2)
        
        
        self.get_abstract_caps2 = Get_Abstract_Caps_Down(self.batch_size,
                                                         capsin_n_maps = 10,
                                                         capsin_n_dims = 5,
                                                         capsout_n_maps = 15,
                                                         capsout_n_dims = 8,
                                                         old_h = 64,
                                                         old_w = 64,
                                                         y_kernel = 2,
                                                         x_kernel = 2)
        
        self.get_abstract_caps3 = Get_Abstract_Caps_Down(self.batch_size,
                                                         capsin_n_maps = 15,
                                                         capsin_n_dims = 8,
                                                         capsout_n_maps = 20,
                                                         capsout_n_dims = 10,
                                                         old_h = 32,
                                                         old_w = 32,
                                                         y_kernel = 4,
                                                         x_kernel = 4)
        
        self.get_abstract_caps_bot = Get_Abstract_Caps_Down(self.batch_size,
                                                            capsin_n_maps = 20,
                                                            capsin_n_dims = 10,
                                                            capsout_n_maps = 20,
                                                            capsout_n_dims = 10,
                                                            old_h = 8,
                                                            old_w = 8,
                                                            y_kernel = 1,
                                                            x_kernel = 1)
                
        self.get_abstract_caps3u = Get_Abstract_Caps_Up(self.batch_size,
                                                        capsin_n_maps = 20 + 20,
                                                        capsin_n_dims = 10,
                                                        capsout_n_maps = 15,
                                                        capsout_n_dims = 8,
                                                        old_h = 8,
                                                        old_w = 8,
                                                        y_kernel = 4,
                                                        x_kernel = 4)

        self.get_abstract_caps2u = Get_Abstract_Caps_Up(self.batch_size,
                                                        capsin_n_maps = 15 + 15,
                                                        capsin_n_dims = 8,
                                                        capsout_n_maps = 10,
                                                        capsout_n_dims = 5,
                                                        old_h = 32,
                                                        old_w = 32,
                                                        y_kernel = 2,
                                                        x_kernel = 2)        

        self.get_abstract_caps1u = Get_Abstract_Caps_Up(self.batch_size,
                                                        capsin_n_maps = 10 + 10,
                                                        capsin_n_dims = 5,
                                                        capsout_n_maps = 8,
                                                        capsout_n_dims = 3,
                                                        old_h = 64,
                                                        old_w = 64,
                                                        y_kernel = 2,
                                                        x_kernel = 2)
        
        
        self.get_abstract_caps_final = Get_Abstract_Caps_Up(self.batch_size,
                                                            capsin_n_maps = 8 + 8,
                                                            capsin_n_dims = 3,
                                                            capsout_n_maps = 3,
                                                            capsout_n_dims = 2,
                                                            old_h = 128,
                                                            old_w = 128,
                                                            y_kernel = 4,
                                                            x_kernel = 4)
        
        #self.class_predictor = ClassPredictor(threshold = 0.6)
        
    def forward(self, x):
        x = self.get_prim_caps(x)
        x_prim = x
        #print(x.size(),'0')
        
        x = self.get_abstract_caps1(x)
        x_1 = x
        #print(x.size(), '1')
        
        x = self.get_abstract_caps2(x)
        x_2 = x
        #print(x.size(), '2')
        
        x = self.get_abstract_caps3(x)
        x_3 = x
        #print(x.size(), '3')
        
        
        x = self.get_abstract_caps_bot(x)
        #x_bot = x
        #print(x.size(), 'bot')
        
        #gotta be careful on the way up there are double maps
        x = torch.cat((x, x_3), 1)
        x = self.get_abstract_caps3u(x)
        #print(x.size(), '3u')
        
        x = torch.cat((x, x_2), 1)
        x = self.get_abstract_caps2u(x)
        #print(x.size(), '2u')
        
        x = torch.cat((x, x_1), 1)
        x = self.get_abstract_caps1u(x)
        #print(x.size(), '1u')
        
        x = torch.cat((x, x_prim), 1)
        x = self.get_abstract_caps_final(x)
        #print(x.size(), '1u')
        
        #x = self.class_predictor(x)
        x = safe_norm(x)
        
        
        return x


###############################################################################
        
def safe_norm(s, axis=-1, epsilon=1e-7):
    squared_norm = torch.mul(s,s).sum(dim=axis)
    return torch.sqrt(squared_norm + epsilon)

###############################################################################
class Dice_Loss(torch.nn.Module):
    """This is a custom Dice Similarity Coefficient loss function that we use 
    to the accuracy of the segmentation. it is defined as ;
    DSC = 2 * (pred /intersect label) / (pred /union label) for the loss we use
    1- DSC so gradient descent leads to better outputs."""
    def __init__(self, weight=None, size_average=False):
        super(Dice_Loss, self).__init__()
        
    def forward(self, pred, label):
        smooth = 1.              #helps with backprop
        intersection = torch.sum(pred * label)
        union = torch.sum(pred) + torch.sum(label)
        loss = (2. * intersection + smooth) / (union + smooth)
        #return 1-loss because we want to minimise dissimilarity
        return 1 - (loss)
