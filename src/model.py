#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:07:35 2019

@author: arjunbalaji
"""
#caps net idea for oct lumen profiler 
#hi
#this iteration of code runs a conv capsule idea

import numpy as np
import torch 

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
class Get_Primary_Caps(torch.nn.Module):
    """This is the primary caps block. It takes in an input image of 1 channel
    and 512x512 pixels and outputs a caps1_n_caps primary caps each which is a
    caps1_n_dims dimensional vector. there is work to be done here so that 
    the numbers all make themselves work. at the moment you have to carefully 
    check each one to make sure the model runs"""
    def __init__(self,
                 input_channels,
                 caps1_n_maps,
                 caps1_caps_grid_ydim,
                 caps1_caps_grid_xdim,
                 caps1_n_dims):
        super(Get_Primary_Caps, self).__init__()
        self.input_channels = input_channels
        self.caps1_n_maps = caps1_n_maps 
        self.caps1_caps_grid_ydim = caps1_caps_grid_ydim
        self.caps1_caps_grid_xdim = caps1_caps_grid_xdim
        self.caps1_n_dims = caps1_n_dims 
        
        self.relu = torch.nn.ReLU()
        
        #these must be calculated from the input shape and the convs we use!!!
        #this is so important dont get it wrong 
        #O = (W - K - 2P)/S +1
        
        conv1_parameters = {'i': self.input_channels, 'o': 32, 'k': 5, 's': 2, 'p':2}
        self.conv1 = torch.nn.Conv2d(in_channels=conv1_parameters['i'],
                                     out_channels=conv1_parameters['o'],
                                     kernel_size=conv1_parameters['k'],
                                     stride=conv1_parameters['s'],
                                     padding=conv1_parameters['p'])
        
        
        conv2_parameters = {'i': 32, 'o': self.caps1_n_dims * self.caps1_n_maps, 'k': 5, 's': 2, 'p':2}
        self.conv2 = torch.nn.Conv2d(in_channels=conv2_parameters['i'],
                                     out_channels=conv2_parameters['o'],
                                     kernel_size=conv2_parameters['k'],
                                     stride=conv2_parameters['s'],
                                     padding=conv2_parameters['p'])
        
            
    def output_params(self):
        return {'maps out': self.caps1_n_maps,
                'caps dim': self.caps1_n_dims,
                'h': self.caps1_caps_grid_ydim,
                'w': self.caps1_caps_grid_xdim}
    
    def forward(self, x):
        #print(x.size())
        x = self.conv1(x)
        #print(x.size())
        x = self.relu(x)
        x = self.conv2(x)
        #print(x.size())
        x = self.relu(x)
        
        '''
        THIS IS IMPORTANT !!!!!!!!!!!!!!!!!!
        if we just went ;
        x = x.view([-1,
                    self.caps1_n_maps,
                    self.caps1_caps_grid_ydim, 
                    self.caps1_caps_grid_xdim,
                    self.caps1_n_dims])
        then pytorch replicates the images and you get this weird ass grid. 
        so instead we view it as below, then transpose the axes till we get 
        what we want
        '''
        x = x.view([-1,
                    self.caps1_n_maps,
                    self.caps1_n_dims,
                    self.caps1_caps_grid_ydim, 
                    self.caps1_caps_grid_xdim])
        #x = torch.transpose(x, 3, 4)
        #x = torch.transpose(x, 2, -1)
        x = x.permute(0, 1, 3, 4, 2)
        #x = squash(x)
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
                 x_kernel,
                 stride,
                 padding,
                 across = True):
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
        self.stride = stride
        self.padding = padding
        self.across = across
        '''
        #this is our weight matrix that goes into our conv layer to get 
        #prediction vectors
        self.W = torch.nn.Parameter(torch.Tensor(self.capsout_n_maps * self.capsout_n_dims,
                                                      self.capsin_n_maps,
                                                      self.y_kernel,
                                                      self.x_kernel))
        '''
        #this is a SPECIAL BIAS! it doesnt go into our conv opertaion
        #we replicate it later and use it for the caps
        self.bias = torch.nn.Parameter(torch.Tensor(1,
                                                    self.capsout_n_maps,
                                                    1,
                                                    1,
                                                    self.capsout_n_dims)) 
        
        self.reset_parameters()
        
        #self.predict_vectors = Predict_Vectors_Down(self.W, self.bias)
        self.capsconv2d_down = torch.nn.Conv2d(in_channels = self.capsin_n_maps*self.capsin_n_dims,
                                          out_channels = self.capsout_n_maps*self.capsout_n_dims,
                                          kernel_size = (self.y_kernel, self.x_kernel),
                                          stride = self.stride,
                                          padding = self.padding,
                                          bias = True)
        '''
        if self.across:
            self.capsconv2d_across = torch.nn.Conv2d(in_channels = self.capsout_n_maps*self.capsout_n_dims,
                                                     out_channels = self.capsin_n_maps*self.capsout_n_maps*self.capsout_n_dims,
                                                     kernel_size = 1,
                                                     stride = 1,
                                                     padding = 0,
                                                     bias = False)
        '''
        
        #calculate final prediction heights and widths for routing
        self.new_hl = (self.old_h - self.y_kernel + 2*self.padding)/self.stride + 1
        #self.new_hl = (self.new_hl - 1 + 0) / 1 + 1
        
        self.new_wl = (self.old_w - self.x_kernel + 2*self.padding)/self.stride + 1
        #self.new_wl = (self.new_wl - 1 + 0) / 1 + 1
        
        #init routing algorithm
        self.routing = Agreement_Routing_Down(bias=self.bias,
                                              input_caps_maps = self.capsin_n_maps,
                                              input_caps_dim = self.capsin_n_dims,
                                              output_caps_maps = self.capsout_n_maps,
                                              output_caps_dim = self.capsout_n_dims,
                                              new_hl = self.new_hl,
                                              new_wl = self.new_wl,
                                              num_iterations = 2)

    def infer_shapes(self):
        return {'caps maps': self.capsout_n_maps,
                'caps dims': self.capsout_n_dims,
                'h': self.new_hl,
                'w': self.new_wl}
    
    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.capsin_n_maps)
        #self.W.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        #print('no we doing abstract caps1')
        #print(self.W.size(), 'weight size')
        #print(self.bias.size(), 'bias size')
        #print(x.size(), 'caps before conv capsule preds' )
        #x = self.predict_vectors(primary_caps)
        #print(self.new_hl)
        batch, input_maps, hold, wold, input_capdims = x.size()
        
        #these bits are becaause of the view 'replication' issue i was having 
        #print(batch, input_maps, hold, wold, input_capdims)
        #x = torch.transpose(x, -1, 2)
        #x = torch.transpose(x, 3, 4)
        x = x.permute(0, 1, 4, 2, 3)
        #print(x.size())
        
        x = x.contiguous().view([-1,
                    input_maps * input_capdims,
                    hold,
                    wold])
    
        x = self.capsconv2d_down(x)
        #print(x.size(), 'after down conv')
        
        #if self.across:
        #    x = self.capsconv2d_across(x)
            #print(x.size(), 'after across conv')
        
        _, _, hnew, wnew = x.size()
        
        #view issue
        x = x.view([batch,
                    self.capsout_n_maps,
                    self.capsout_n_dims,
                    hnew,
                    wnew,
                    -1])
    
        x = x.permute(0, 1, 3, 4, 5, 2)
        #print(x.size())
        '''
        x = x.view([batch,
                    self.capsout_n_maps,
                    hnew,
                    wnew,
                    -1,
                    self.capsout_n_dims])
        '''
        #print(x.size(), 'resizing to normal')
        
        x = self.routing(x)
        
        return x

###############################################################################

class Agreement_Routing_Down(torch.nn.Module):
    """This is the localised agreement routing algorithm. It takes in the total
    prediction vectors from a layer l and computes the routing weights for 
    those predictions. It then squashes the prediction vectors using the 
    custom squash function."""
    def __init__(self, bias,
                 input_caps_maps,
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
        #self.bias = bias.repeat((1, 1, self.new_hl, self.new_wl, 1))
        
        self.b = torch.nn.Parameter(torch.zeros((1,
                                                 self.output_caps_maps,
                                                 self.new_hl,
                                                 self.new_wl,
                                                 self.input_caps_maps))) #  , device = 'cuda:0')
        
        #self.bias = self.bias.repeat((1, 1, self.new_hl, self.new_wl, 1))
        
    def forward(self, tensor_of_prediction_vector):
        #print('now we doing routing')
        c = self.softmax(self.b)
        
        #print(c.size(),'c')
        #print(tensor_of_prediction_vector.size(), 'pred vectors')
        
        
        #output_vectors = torch.mul(tensor_of_prediction_vector, c.unsqueeze(-1)) 
        output_vectors = torch.mul(c.unsqueeze(-1), tensor_of_prediction_vector) 
        
        #print(output_vectors.size(), 'output vectors after *c')
        #print(self.bias.size(), 'repeated bias size')
        
        output_vectors = output_vectors.sum(dim=-2) #+ self.bias
        
        #print(output_vectors.size(), 'output vectors after sum')
        
        #print(output_vectors.size())
        output_vectors = squash(output_vectors, axis = -1)
        #print(output_vectors.size(), 'should be good')
        b_batch = self.b
        
        #print('routing loop')
        for d in range(self.num_iterations):
            
            #print('preds', tensor_of_prediction_vector.size())
            #print('out', output_vectors.size())
            
            b_batch = b_batch + torch.mul(tensor_of_prediction_vector,
                                output_vectors.unsqueeze(-2)).sum(dim = -1)
            '''
            distances = torch.mul(tensor_of_prediction_vector,
                                output_vectors.unsqueeze(-2)).sum(dim = -1)
            
            self.b = torch.add(self.b, distances)
            '''
            #b_batch = b_batch.sum(dim = -1)
            
            #b_batch = 
            #print('bbatch', b_batch.size())
            
            c = self.softmax(b_batch)
            #print(c.size())
            output_vectors = torch.mul(tensor_of_prediction_vector,
                                       c.unsqueeze(-1))
            
            output_vectors = output_vectors.sum(-2)
            output_vectors = squash(output_vectors,
                                    axis = -1)
            #print(c.size())
        #print(tensor_of_prediction_vector.size())
        #print(output_vectors.size())
        self.c = c
        
        return output_vectors
###############################################################################
        
    
class Get_Abstract_Caps_Up(torch.nn.Module):
    """This is the abstract caps layer. We take in an input of the capsules
    of the previous layer and then output predictions of abstract capsules.
    
        notes:
            . padding must be tuple (x,y)
            . if uptype == 'upsample' used 
                > padding is autocalculated, even with user input
                > stride = 1 is set, even with user stride != 1
            . 
            . output_padding only required if uptype == 'deconv' 
            . output_padding, if used, must be tuple (x,y)
                """
    def __init__(self, 
                 batch_size,
                 capsin_n_maps,
                 capsin_n_dims,
                 capsout_n_maps,
                 capsout_n_dims,
                 old_h,
                 old_w,
                 y_kernel,
                 x_kernel,
                 stride,
                 padding, #MUST E A TUPLE CUNNY
                 output_padding, #outpadding must be a TUPLE BIATCH, also i
                 uptype,
                 across = True):
        super(Get_Abstract_Caps_Up, self).__init__()
        
        self.batch_size = batch_size
        self.capsin_n_dims = capsin_n_dims
        self.capsin_n_maps = capsin_n_maps 
        self.capsout_n_maps = capsout_n_maps 
        self.capsout_n_dims = capsout_n_dims
        
        self.old_h = old_h
        self.old_w = old_w
        self.y_kernel = y_kernel
        self.x_kernel = x_kernel
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.uptype = uptype
        self.across = across
        
        #this is a SPECIAL BIAS! it doesnt go into our conv opertaion
        #we replicate it later and use it for the caps
        self.bias = torch.nn.Parameter(torch.Tensor(1,
                                                    self.capsout_n_maps,
                                                    1,
                                                    1,
                                                    self.capsout_n_dims)) 
        
        self.reset_parameters()
        
        pady = self.padding[0]
        padx = self.padding[1]
        

        
        if self.uptype == 'deconv':
            '''
            HUGELY IMPORTANT TO GET DECONV TO WORK WE ADD 1 to the end
            '''
            self.new_hl = (self.old_h-1)*self.stride + self.y_kernel - 2*pady + self.output_padding[0] 
            self.new_wl = (self.old_w-1)*self.stride + self.x_kernel - 2*padx + self.output_padding[1] 
    
            self.capsconv2d_up = torch.nn.ConvTranspose2d(in_channels = self.capsin_n_maps*self.capsin_n_dims,
                                          out_channels = self.capsout_n_maps*self.capsout_n_dims,
                                          kernel_size = (self.y_kernel, self.x_kernel),
                                          stride = self.stride,
                                          padding = self.padding,
                                          output_padding = self.output_padding,
                                          bias = True)
            
        elif self.uptype == 'upsample':
            #pady = int((self.y_kernel - 1)/2)
            #padx = int((self.x_kernel - 1)/2)
            #self.padding = (pady, padx)
            self.new_hl = 2*self.old_h
            self.new_wl = 2*self.old_w
            self.capsconv2d_up = torch.nn.Conv2d(in_channels = self.capsin_n_maps*self.capsin_n_dims,
                                          out_channels = self.capsout_n_maps*self.capsout_n_dims,
                                          kernel_size = (self.y_kernel, self.x_kernel),
                                          stride = 1,
                                          padding = self.padding,
                                          bias = True)
        

        #init routing algorithm
        self.routing = Agreement_Routing_Down(bias=self.bias,
                                              input_caps_maps = self.capsin_n_maps,
                                              input_caps_dim = self.capsin_n_dims,
                                              output_caps_maps = self.capsout_n_maps,
                                              output_caps_dim = self.capsout_n_dims,
                                              new_hl = self.new_hl,
                                              new_wl = self.new_wl,
                                              num_iterations = 2)

    def infer_shapes(self):
        return {'caps maps': self.capsout_n_maps,
                'caps dims': self.capsout_n_dims,
                'h': self.new_hl,
                'w': self.new_wl}
    
    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.capsin_n_maps)
        #self.W.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        #print('no we doing abstract caps2u')
        
        #print(self.bias.size(), 'bias size')
        #print(x.size(), 'caps before conv capsule preds' )
        #x = self.predict_vectors(primary_caps)
        #print(self.new_hl)
        batch, input_maps, hold, wold, input_capdims = x.size()
        
        x = x.permute(0, 1, 4, 2, 3)
        #print(x.size())
        
        x = x.contiguous().view([-1,
                    input_maps * input_capdims,
                    hold,
                    wold])
        '''
        x = x.view([-1,
                    input_maps * input_capdims,
                    hold,
                    wold])
        '''    
        if self.uptype == 'deconv':
            x = self.capsconv2d_up(x)
            #print(x.size(), 'after up deconv')
        
        elif self.uptype == 'upsample':
            #print(x.size(),'pre up')
            x = torch.nn.functional.upsample(x,
                                             size=[self.new_hl, self.new_wl], 
                                             mode='bilinear')
            #print(x.size())
            x = self.capsconv2d_up(x)
        
        
        
        
        
        '''
        if self.across:
            x = self.capsconv2d_across(x)
            #print(x.size(), 'after across conv')
        '''
        
        #just 
        _, _, hnew, wnew = x.size()
        
        if self.new_hl != hnew: 
            print('Something funny going on with user defined hnew and actual')
            
        
        if self.new_wl != wnew: 
            print('Something funny going on with user defined wnew and actual')
            
        #x = x.permute(0, 1, 3, 4, 5, 2)
        x = x.view([batch,
                    self.capsout_n_maps,
                    self.capsout_n_dims,
                    hnew,
                    wnew,
                    -1])
    
        x = x.permute(0, 1, 3, 4, 5, 2)
        
        #print(x.size(), 'resizing to normal')
        
        x = self.routing(x)
        #print(x.size(), 'out')
        return x

###############################################################################
        
class Reconstruction_Layer(torch.nn.Module):
    """TThis is the reconstruction layer for the network to learn how to remake
    the original input image"""
    def __init__(self, 
                 batch_size,
                 capsin_n_maps,
                 capsin_n_dims,
                 reconstruct_channels):
        super(Reconstruction_Layer, self).__init__()
        
        self.batch_size = batch_size
        self.capsin_n_dims = capsin_n_dims
        self.capsin_n_maps = capsin_n_maps 
        self.reconstruct_channels = reconstruct_channels
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        
        self.conv1_params = {'i':int(self.capsin_n_maps*self.capsin_n_dims),
                             'o':64,
                             'k':1,
                             's':1,
                             'p':0}
        self.conv1 = torch.nn.Conv2d(in_channels = self.conv1_params['i'],
                                     out_channels = self.conv1_params['o'],
                                     kernel_size = self.conv1_params['k'],
                                     stride = self.conv1_params['s'],
                                     padding = self.conv1_params['p'])
        
        self.conv2_params = {'i':int(self.conv1_params['o']),
                             'o':128,
                             'k':1,
                             's':1,
                             'p':0}
        self.conv2 = torch.nn.Conv2d(in_channels = self.conv2_params['i'],
                                     out_channels = self.conv2_params['o'],
                                     kernel_size = self.conv2_params['k'],
                                     stride = self.conv2_params['s'],
                                     padding = self.conv2_params['p'])
        
        self.conv3_params = {'i':int(self.conv2_params['o']),
                             'o':self.reconstruct_channels,
                             'k':1,
                             's':1,
                             'p':0}
        self.conv3 = torch.nn.Conv2d(in_channels = self.conv3_params['i'],
                                     out_channels = self.conv3_params['o'],
                                     kernel_size = self.conv3_params['k'],
                                     stride = self.conv3_params['s'],
                                     padding = self.conv3_params['p'])
        
    def forward(self, x):
        
        _, _, h, w, _ = x.size()
        #print(x.size())
        x = x.permute(0, 1, 4, 2, 3)
        #print(x.size())
        x = x.contiguous().view([-1,
                    self.capsin_n_maps * self.capsin_n_dims,
                    h, 
                    w])
        #print(x.size())
        #x = torch.transpose(x, 3, 4)
        #x = torch.transpose(x, 2, -1)
        
        
        x = self.conv1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.sigmoid(x)
        
        return x
###############################################################################    
    
class CapsNet(torch.nn.Module):
    """This is the actual model. It has a down line, a bottom pass and then a 
    series of up passes. On the up passes we concatenate prior down passes as 
    this improves the networks localisation. it is important the the tensors we 
    concatenate are the same size. so we use upsampling. Also be aware of the 
    channels, we want a lot of channels (~1000) so the network learns intricate
    features."""
    def __init__(self, opt):
        super(CapsNet, self).__init__()
        self.opt = opt
        
        if self.opt.transforms:
            caps1_ygrid = self.opt.c_size[0]
            caps1_xgrid = self.opt.c_size[1]
        else:
            caps1_ygrid = self.opt.start_size[0]
            caps1_xgrid = self.opt.start_size[1]      
            
        self.get_prim_caps = Get_Primary_Caps(input_channels=self.opt.inputchannels,
                                              caps1_n_maps = self.opt.primmaps,
                                              caps1_caps_grid_ydim = int(caps1_ygrid / 4),
                                              caps1_caps_grid_xdim = int(caps1_xgrid / 4),
                                              caps1_n_dims = self.opt.primdims)
        prim_params = self.get_prim_caps.output_params()
        
        self.get_abstract_caps1 = Get_Abstract_Caps_Down(self.opt.batch_size,
                                                         capsin_n_maps = prim_params['maps out'],
                                                         capsin_n_dims = prim_params['caps dim'],
                                                         capsout_n_maps = self.opt.maps1,
                                                         capsout_n_dims = self.opt.dims1,
                                                         old_h = prim_params['h'],
                                                         old_w = prim_params['w'],
                                                         y_kernel = 5,
                                                         x_kernel = 5,
                                                         stride = 1,
                                                         padding = 2,
                                                         across=True)
        caps1_params = self.get_abstract_caps1.infer_shapes()
        
        self.get_abstract_caps1a = Get_Abstract_Caps_Down(self.opt.batch_size,
                                                         capsin_n_maps = caps1_params['caps maps'],
                                                         capsin_n_dims = caps1_params['caps dims'],
                                                         capsout_n_maps = caps1_params['caps maps'],
                                                         capsout_n_dims = caps1_params['caps dims'],
                                                         old_h = caps1_params['h'],
                                                         old_w = caps1_params['w'],
                                                         y_kernel = 6,
                                                         x_kernel = 6,
                                                         stride = 2,
                                                         padding = 2,
                                                         across=True)
        caps1a_params = self.get_abstract_caps1a.infer_shapes()
        
        self.get_abstract_caps2 = Get_Abstract_Caps_Down(self.opt.batch_size,
                                                         capsin_n_maps = caps1a_params['caps maps'],
                                                         capsin_n_dims = caps1a_params['caps dims'],
                                                         capsout_n_maps = self.opt.maps2,
                                                         capsout_n_dims =  self.opt.dims2,
                                                         old_h = int(caps1a_params['h']),
                                                         old_w = int(caps1a_params['w']),
                                                         y_kernel = 5,
                                                         x_kernel = 5,
                                                         stride = 1,
                                                         padding = 2,
                                                         across=False)
        caps2_params = self.get_abstract_caps2.infer_shapes()

        self.get_abstract_caps2a = Get_Abstract_Caps_Down(self.opt.batch_size,
                                                         capsin_n_maps = caps2_params['caps maps'],
                                                         capsin_n_dims = caps2_params['caps dims'],
                                                         capsout_n_maps = caps2_params['caps maps'],
                                                         capsout_n_dims = caps2_params['caps dims'],
                                                         old_h = caps2_params['h'],
                                                         old_w = caps2_params['w'],
                                                         y_kernel = 6,
                                                         x_kernel = 6,
                                                         stride = 2,
                                                         padding = 2,
                                                         across=True)
        caps2a_params = self.get_abstract_caps2a.infer_shapes()
        
        self.get_abstract_caps3 = Get_Abstract_Caps_Down(self.opt.batch_size,
                                                         capsin_n_maps = caps2a_params['caps maps'],
                                                         capsin_n_dims = caps2a_params['caps dims'],
                                                         capsout_n_maps = self.opt.maps3,
                                                         capsout_n_dims = self.opt.dims3,
                                                         old_h = int(caps2a_params['h']),
                                                         old_w = int(caps2a_params['w']),
                                                         y_kernel = 5,
                                                         x_kernel = 5,
                                                         stride = 1,
                                                         padding = 2,
                                                         across=False)
        caps3_params = self.get_abstract_caps3.infer_shapes()

        self.get_abstract_caps3a = Get_Abstract_Caps_Down(self.opt.batch_size,
                                                         capsin_n_maps = caps3_params['caps maps'],
                                                         capsin_n_dims = caps3_params['caps dims'],
                                                         capsout_n_maps = caps3_params['caps maps'],
                                                         capsout_n_dims = caps3_params['caps dims'],
                                                         old_h = caps3_params['h'],
                                                         old_w = caps3_params['w'],
                                                         y_kernel = 6,
                                                         x_kernel = 6,
                                                         stride = 2,
                                                         padding = 2,
                                                         across=True)
        caps3a_params = self.get_abstract_caps3a.infer_shapes()
        
        self.get_abstract_caps_bot = Get_Abstract_Caps_Down(self.opt.batch_size,
                                                         capsin_n_maps = caps3a_params['caps maps'],
                                                         capsin_n_dims = caps3a_params['caps dims'],
                                                         capsout_n_maps = caps3a_params['caps maps'],
                                                         capsout_n_dims = caps3a_params['caps dims'],
                                                         old_h = int(caps3a_params['h']),
                                                         old_w = int(caps3a_params['w']),
                                                         y_kernel = 1,
                                                         x_kernel = 1,
                                                         stride = 1,
                                                         padding = 0,
                                                         across=True)
        capsbot_params = self.get_abstract_caps_bot.infer_shapes()
        
        if self.opt.uptype == 'upsample':
            upa_outputpadding = (0,0)
        elif self.opt.uptype == 'deconv':
            upa_outputpadding = (1,1)
            
        self.get_abstract_caps3u = Get_Abstract_Caps_Up(self.opt.batch_size,
                                                         capsin_n_maps = capsbot_params['caps maps'] + caps3_params['caps maps'],
                                                         capsin_n_dims = capsbot_params['caps dims'],
                                                         capsout_n_maps = caps2_params['caps maps'],
                                                         capsout_n_dims = caps2_params['caps dims'],
                                                         old_h = int(capsbot_params['h']),
                                                         old_w = int(capsbot_params['w']),
                                                         y_kernel = 5,
                                                         x_kernel = 5,
                                                         stride = 2,
                                                         padding = (2,2),
                                                         output_padding=upa_outputpadding,
                                                         uptype = self.opt.uptype,
                                                         across=True)
        caps3u_params = self.get_abstract_caps3u.infer_shapes()
        
        self.get_abstract_caps3ua = Get_Abstract_Caps_Down(self.opt.batch_size,
                                                         capsin_n_maps = caps3u_params['caps maps'],
                                                         capsin_n_dims = caps3u_params['caps dims'],
                                                         capsout_n_maps = caps3u_params['caps maps'],
                                                         capsout_n_dims = caps3u_params['caps dims'],
                                                         old_h = caps3u_params['h'],
                                                         old_w = caps3u_params['w'],
                                                         y_kernel = 5,
                                                         x_kernel = 5,
                                                         stride = 1,
                                                         padding = 2,
                                                         across=True)
        caps3ua_params = self.get_abstract_caps3ua.infer_shapes()
        
        self.get_abstract_caps2u = Get_Abstract_Caps_Up(self.opt.batch_size,
                                                         capsin_n_maps = caps3ua_params['caps maps'] + caps2_params['caps maps'],
                                                         capsin_n_dims = caps3ua_params['caps dims'],
                                                         capsout_n_maps = caps1_params['caps maps'],
                                                         capsout_n_dims = caps1_params['caps dims'],
                                                         old_h = int(caps3ua_params['h']),
                                                         old_w = int(caps3ua_params['w']),
                                                         y_kernel = 5,
                                                         x_kernel = 5,
                                                         stride = 2,
                                                         padding = (2,2),
                                                         output_padding=upa_outputpadding,
                                                         uptype = self.opt.uptype,
                                                         across=True)
        caps2u_params = self.get_abstract_caps2u.infer_shapes()
        
        self.get_abstract_caps2ua = Get_Abstract_Caps_Down(self.opt.batch_size,
                                                         capsin_n_maps = caps2u_params['caps maps'],
                                                         capsin_n_dims = caps2u_params['caps dims'],
                                                         capsout_n_maps = caps2u_params['caps maps'],
                                                         capsout_n_dims = caps2u_params['caps dims'],
                                                         old_h = caps2u_params['h'],
                                                         old_w = caps2u_params['w'],
                                                         y_kernel = 5,
                                                         x_kernel = 5,
                                                         stride = 1,
                                                         padding = 2,
                                                         across=True)
        caps2ua_params = self.get_abstract_caps2ua.infer_shapes()
        
        self.get_abstract_caps1u = Get_Abstract_Caps_Up(self.opt.batch_size,
                                                         capsin_n_maps = caps2ua_params['caps maps'] + caps1_params['caps maps'],
                                                         capsin_n_dims = caps2ua_params['caps dims'],
                                                         capsout_n_maps = prim_params['maps out'],
                                                         capsout_n_dims = prim_params['caps dim'],
                                                         old_h = int(caps2ua_params['h']),
                                                         old_w = int(caps2ua_params['w']),
                                                         y_kernel = 5,
                                                         x_kernel = 5,
                                                         stride = 2,
                                                         padding = (2,2),
                                                         output_padding=upa_outputpadding,
                                                         uptype = self.opt.uptype,
                                                         across=True)
        caps1u_params = self.get_abstract_caps1u.infer_shapes()
        

        self.get_abstract_caps1ua = Get_Abstract_Caps_Down(self.opt.batch_size,
                                                         capsin_n_maps = caps1u_params['caps maps'],
                                                         capsin_n_dims = caps1u_params['caps dims'],
                                                         capsout_n_maps = caps1u_params['caps maps'],
                                                         capsout_n_dims = caps1u_params['caps dims'],
                                                         old_h = caps1u_params['h'],
                                                         old_w = caps1u_params['w'],
                                                         y_kernel = 5,
                                                         x_kernel = 5,
                                                         stride = 1,
                                                         padding = 2,
                                                         across=True)
        caps1ua_params = self.get_abstract_caps1ua.infer_shapes()
        
        self.get_abstract_caps_final1 = Get_Abstract_Caps_Up(self.opt.batch_size,
                                                         capsin_n_maps = caps1ua_params['caps maps'] + prim_params['maps out'],
                                                         capsin_n_dims = caps1ua_params['caps dims'],
                                                         capsout_n_maps = self.opt.f1maps,
                                                         capsout_n_dims = self.opt.f1dims,
                                                         old_h = int(caps1ua_params['h']),
                                                         old_w = int(caps1ua_params['w']),
                                                         y_kernel = 7,
                                                         x_kernel = 7,
                                                         stride = 2,
                                                         padding = (3,3),
                                                         output_padding=upa_outputpadding,
                                                         uptype = self.opt.uptype,
                                                         across=True)
        capsfinal1_params = self.get_abstract_caps_final1.infer_shapes()
        
        self.get_abstract_caps_final2 = Get_Abstract_Caps_Up(self.opt.batch_size,
                                                         capsin_n_maps = self.opt.f1maps,
                                                         capsin_n_dims = self.opt.f1dims,
                                                         capsout_n_maps = self.opt.f2maps,
                                                         capsout_n_dims = self.opt.f2dims,
                                                         old_h = int(capsfinal1_params['h']),
                                                         old_w = int(capsfinal1_params['w']),
                                                         y_kernel = 7,
                                                         x_kernel = 7,
                                                         stride = 2,
                                                         padding = (3,3),
                                                         output_padding=upa_outputpadding,
                                                         uptype = self.opt.uptype,
                                                         across=False)

        self.reconstruct = Reconstruction_Layer(self.opt.batch_size,
                                                capsin_n_maps = self.opt.f2maps,
                                                capsin_n_dims = self.opt.f2dims,
                                                reconstruct_channels=self.opt.reconchannels)
        
    def forward(self, x):
        x = self.get_prim_caps(x)
        x_prim = x
        #print(x.size(),'0')
        self.x_prim = x_prim
        #print('##########################FINISHED PRIM#######################')
        x = self.get_abstract_caps1(x)
        
        #print(x.size(), '1')
        x = self.get_abstract_caps1a(x)
        x_1 = x
        #print(x.size(), '1a')
        self.x_1 = x_1
        #print('##########################FINISHED 1#######################')
        
        x = self.get_abstract_caps2(x)
        #print(x.size(), '2')
        x = self.get_abstract_caps2a(x)
        x_2 = x
        #print(x.size(), '2a')
        self.x_2 = x_2
        
        x = self.get_abstract_caps3(x)
        #print(x.size(), '3')
        x = self.get_abstract_caps3a(x)
        x_3 = x
        #print(x.size(), '3a')
        self.x_3 = x_3
             
        x = self.get_abstract_caps_bot(x)
        #x_bot = x
        #print(x.size(), 'bot')
        
        #gotta be careful on the way up there are double maps
        x = torch.cat((x, x_3), 1)
        
        x = self.get_abstract_caps3u(x)
        #print(x.size(), '3u')
        x = self.get_abstract_caps3ua(x)
        #print(x.size(), '3ua')
        

        x = torch.cat((x, x_2), 1)
        x = self.get_abstract_caps2u(x)
        #print(x.size(), '2u')
        x = self.get_abstract_caps2ua(x)
        #print(x.size(), '2ua')
        
        x = torch.cat((x, x_1), 1)
        x = self.get_abstract_caps1u(x)
        #print(x.size(), '1u')
        x = self.get_abstract_caps1ua(x)
        #print(x.size(), '1ua')

        x = torch.cat((x, x_prim), 1)
        x = self.get_abstract_caps_final1(x)
        #print(x.size(), 'final1')

        x = self.get_abstract_caps_final2(x)
        #print(x.size(), 'final2')
        reconstruct = self.reconstruct(x)
        
        x = safe_norm(x)
        
        #print(x.size(), 'last')
        return x, reconstruct


###############################################################################
        
def safe_norm(s, axis=-1, epsilon=1e-7):
    squared_norm = torch.mul(s,s).sum(dim=axis)
    return torch.sqrt(squared_norm + epsilon)

###############################################################################
class Dice_Loss(torch.nn.Module):
    """This is a custom Dice Similarity Coefficient loss function that we use 
    to the accuracy of the segmentation. it is defined as ;
    DSC = 2 * (pred /intersect label) / (pred /union label) for the losss we use
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

'''
 def printskeet(i):
    f, (ax1i, ax1df, ax1l) = plt.subplots(1,3, sharey=True)
    pyplot.tight_layout()

    #raw image
    ax1i.imshow(allimages[i,:,:],
           aspect = 'equal')
    #pred
    ax1df.imshow(alldf[i,:,:],
           aspect = 'equal')
    #pred after thresholding
    ax1l.imshow(alllabels[i,:,:],
                   aspect = 'equal')
    #original label

    f.colorbar(ax1i.imshow(allimages[i,:,:], 
                      aspect = 'equal'))
    pyplot.show()
'''