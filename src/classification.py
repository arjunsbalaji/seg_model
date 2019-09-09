#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:23:20 2019

@author: arjun
"""

import numpy as np
import torch, os, sys, time, argparse, json
from model import *
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import fastai.vision as fv
from skimage.filters import gaussian
from sklearn import preprocessing
import skimage.transform as skitransforms
from options import OptionsHome


starting = time.time()

class RandomSingleImageCrop(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        
        
    def __call__(self, image):

        
        h, w, _ = image.shape
        
        new_h, new_w = self.output_size
        
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        
        image = image[top:top + new_h, left:left + new_w, :]
        #label = label[top:top + new_h, left:left + new_w, :]
    
        return image#, label

def get_image(main_data_dir, name):
    this_data_path = os.path.join(main_data_dir)
    return np.load(os.path.join(this_data_path, name))

class OCTClassificationDataset(Dataset):
    """
    First we create a dataset that will encapsulate our data. It has 3 special 
    functions which will be explained as they go. We will pass this dataset object
    to the torch dataloader object later which will make training easier.
    """
    def __init__(self,
                  image_data_dir,
                  labels_dict,
                  start_size,
                  cropped_size,
                  transform):
        self.image_data_dir = image_data_dir
        self.labels_dict = labels_dict 
        self.start_size = start_size
        self.transform = transform
        self.cropped_size = cropped_size
        
        
        self.rcrop = RandomSingleImageCrop(self.cropped_size)
        self.phflip = np.random.rand()
        self.pvflip = np.random.rand()
        
        #iterate through the 2d images and get all their name
            
        self.name_list = list(self.labels_dict.keys())
    
    def visualise3(self, idx):
        
        
        sample = self.__getitem__(idx)
        #print(sample['input'].size())
        #print(sample['label'].size())
        input_data = sample['input'].cpu().numpy()[0,:,:]
        label = sample['label']
        
        if label==0:
            label='easy'
        else:
            label='hard'
        
        
        f, (ax1) = plt.subplots(1,1, sharey=True)
        f.subplots_adjust(hspace=0.3)
        plt.tight_layout()
        
        #plot image
        image1 = ax1.imshow(input_data,
                            aspect = 'equal')
        
        f.colorbar(image1, ax=ax1, orientation='vertical', fraction = 0.05)
        ax1.set_title(label)
        
        
        plt.show()
        
    def __getitem__(self, idx):
        """This function will allow us to index the data object and it will 
        return a sample."""
        name = self.name_list[idx]
        
        #load data  
        label = self.labels_dict[name]
        
        image = get_image(self.image_data_dir, name)
        #image = np.expand_dims(image[0,:,:],0)
        image = image.astype(float)
        
        
        #print(image.shape)
        
        image = np.transpose(image, (1, 2, 0))
        #print(label.max())
        #print(Image.shape)
        if self.transform:
            
            ysize = self.start_size[0] + 20
            xsize = self.start_size[1] + 20
            image = skitransforms.resize(image, output_shape=(ysize, xsize))          
            
            #print(label.shape)
            #print(label.max())
            image = self.rcrop(image)
            #print(label.max())
            
            if self.phflip>0.5:
                #hflip
                image = np.flip(image, 1)
                #print(label.max())
            #print(label.shape)
            
            if self.pvflip>0.5:
                #vflip
                image = np.flip(image, 0)
                #print(label.max())
            #print(label.shape)
            
            angle = np.random.randint(0,360)
            image = skitransforms.rotate(image, angle=angle, mode='reflect')
            #print(label.max())
            #print(label.shape)
            
            if np.random.rand() > 0.5:
                image = gaussian(image, sigma=1, mode='reflect')
            
            
        else:
            image = skitransforms.resize(image, output_shape= self.start_size)
        
        #image = np.expand_dims(preprocessing.scale(image[:,:,0]), -1)
        
        image = np.transpose(image.copy(), (2, 0, 1))
        #og = preprocessing.MinMaxScaler(og)
        
        sample = {'input': torch.tensor(image).float(),
                  'label': torch.tensor(label),
                  'case_name': name}

        return sample
    
    def __len__(self):    
        """This function is mandated by Pytorch and allows us to see how many 
        data points we have in our dataset"""
        return len(self.name_list)
    
f = open("/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/Final1-pawsey/analysis/softdicepairs.json")
softdice = json.load(f)
f.close()

for k,v in softdice.items():
    if v<0.96:
        softdice[k] = 1 # 9.3656 is class imbalance coefficient
    else:
        softdice[k] = 0

#all the ones that are hard
hardnames=[]
for k,v in softdice.items():
    if v!=0:
        hardnames.append(k)



o = OptionsHome()
o.parse()

image_dir = fv.Path('/media/arjun/VascLab EVO/projects/oct_ca_seg/actual final data/images')

oct_class_dataset = OCTClassificationDataset(image_data_dir=image_dir,
                                             labels_dict=softdice,
                                             start_size=o.opt.start_size,
                                             cropped_size=o.opt.c_size,
                                             transform=True)

hardindices = [oct_class_dataset.name_list.index(i) for i in hardnames]

class LinReluD(torch.nn.Module):
    """New linear block for my classification network"""
    def __init__(self, Fin, Fout):#, first=True):
        super(LinReluD, self).__init__()
        self.Fin = Fin
        self.Fout = Fout
        #self.first = first
        
        self.lin = torch.nn.Linear(self.Fin, self.Fout)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout2d(p=0.2)
        self.batchN = torch.nn.BatchNorm1d(self.Fout)
        #self.lin1 = torch.nn.Linear(self.Fout, self.Fout)
        
    def forward(self, x):
        x = self.lin(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return x
    

class ClassifyCapsNet(torch.nn.Module):
    '''Classification net that uses downsampling pathway of Capsnet for transfer learning!'''
    def __init__(self, opt):
        super(ClassifyCapsNet, self).__init__()
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
        
        #lin1in = int(self.opt.batch_size * np.prod(list(capsbot_params.values())))
        lin1in = int(np.prod(list(capsbot_params.values())))
        self.lin1 = LinReluD(lin1in, 1000)
        self.lin2 = LinReluD(1000, 1000)
        self.lin3 = LinReluD(1000, 2)
        
    def forward(self, x):
        
        inbatch=x.size()[0]
        x = self.get_prim_caps(x)
        #x_prim = x
        #print(x.size(),'0')
        #self.x_prim = x_prim
        #print('##########################FINISHED PRIM#######################')
        x = self.get_abstract_caps1(x)
        
        #print(x.size(), '1')
        x = self.get_abstract_caps1a(x)
        #x_1 = x
        #print(x.size(), '1a')
        #self.x_1 = x_1
        #print('##########################FINISHED 1#######################')
        
        x = self.get_abstract_caps2(x)
        #print(x.size(), '2')
        x = self.get_abstract_caps2a(x)
        #x_2 = x
        #print(x.size(), '2a')
        #self.x_2 = x_2
        
        x = self.get_abstract_caps3(x)
        #print(x.size(), '3')
        x = self.get_abstract_caps3a(x)
        #x_3 = x
        #print(x.size(), '3a')


        x = self.get_abstract_caps_bot(x)
        
        x = x.view([inbatch, -1])
        #print(x.size())
        x = self.lin1(x)
        #print(x.size())
        x = self.lin2(x)
        #print(x.size())
        x = self.lin3(x)
        #print(x.size())        
        return x


transfer_model_dict= torch.load('/media/arjun/VascLab EVO/projects/oct_ca_seg/runsaves/Final1-pawsey/checkpoints/checkpoint.pt')['model_state_dict']

cappy = ClassifyCapsNet(o.opt)
cappy = cappy.to(o.opt.device)
cappy.load_state_dict(transfer_model_dict, strict=False)

traind, vald = torch.utils.data.random_split(oct_class_dataset, [1922,481])

trainloady=DataLoader(traind, batch_size=int(o.opt.batch_size), shuffle=True)

valloady = DataLoader(vald, batch_size=1, shuffle=False)

cross = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(cappy.parameters(), lr=0.001)

trainingloss = []
valloss = {}
for epoch in range(1):
    for i, sample in enumerate(trainloady):
        inny = sample['input'].to(o.opt.device)
        label = sample['label'].to(o.opt.device)
        
        optim.zero_grad()
        
        outgvng = cappy(inny)
        loss = cross(outgvng, label)
        loss.backward()
        trainingloss.append(float(loss.data))
        optim.step()
        
    print('val')
    cappy.eval()    
    for i, sample in enumerate(valloady):
        inny = sample['input'].to(o.opt.device)
        label = sample['label'].to(o.opt.device)
        
        outgvng = cappy(inny)
        pres = torch.argmax(outgvng, 1)
        
        valloss[sample['case_name'][0]]=bool((pres==label).cpu())
        
        
        
    
print(time.time()-starting)

