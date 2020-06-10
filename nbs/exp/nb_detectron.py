#!/usr/bin/env python3

import detectron2

import os, sys, time, random
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
import cv2

from detectron2.data.datasets import load_coco_json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2 import model_zoo
from detectron2.evaluation import DatasetEvaluator,DatasetEvaluators

import matplotlib.pyplot as plt
import torch
import json


def save_cfg(cfg, fp):
    with open(fp, 'w') as file:
        file.write(cfg.dump())

#export
def annsToSingleBinMask(cocoset, img_id):
    anns = cocoset.imgToAnns[img_id]
    if len(anns) == 0:
        h, w = cocoset.imgs[img_id]['height'], cocoset.imgs[img_id]['width']
        return np.zeros((h,w))
    else:
        masks = cocoset.annToMask(anns[0])
        for ann in anns[1:]:
            masks = masks + cocoset.annToMask(ann) #assumption that lumen annotations never overlap
        return masks
    
    #export
def lossdice(c,l, iou:bool=False, eps:float=1e-8):
    "Dice coefficient metric for binary target. If iou=True, returns iou metric, classic for segmentation problems."
    n = l.shape[0]
    c = c.view(n,-1).float()
    l = l.reshape(n, -1) # l.view(n,-1)
    intersect = (c * l).sum().float()
    union = (c+l).sum().float()
    if not iou: return (2. * intersect / union if union > 0 else union.new([1.]).squeeze())
    else: return (intersect / (union-intersect+eps) if union > 0 else union.new([1.]).squeeze())
    

    #export
def Sens(c, l):
    #returns sens of argmaxxed predition. 
    #print(c.size(), l.size())
    n_targs=l.size()[0]
    c =(c.view(n_targs, -1) > 0).float()
    #l=(l.reshape(n, -1) > 0).float()
    l=(l.reshape(n_targs, -1) > 0).float()
    inter = torch.sum(c*l, dim=(1))
    union = torch.sum(c, dim=(1)) + torch.sum(l, dim=1) - inter
    #print(inter.size(), union.size())
    return inter/union

#export
def Spec(c,l):
    #returns sens of argmaxxed predition. 
    n_targs=l.size()[0]
    c =(c.view(n_targs, -1) > 0).float()
    #l=(l.reshape(n, -1) > 0).float()
    l=(l.reshape(n_targs, -1) > 0).float()
    c = 1-c
    l=1-l
    inter = torch.sum(c*l, dim=(1))
    union = torch.sum(c, dim=(1)) + torch.sum(l, dim=1) - inter
    return inter/union

#export
def Acc(c, l):
    n_targs=l.size()[0]
    c =(c.view(n_targs, -1) > 0).float()
    l= (l.view(n_targs, -1) > 0).float()
    l=(l.reshape(n_targs, -1) > 0).float()
    c = torch.sum(torch.eq(c,l).float(),dim=1)
    return (c/l.size()[-1]).mean()

#export
class OCT_Evaluator(DatasetEvaluator):
    def __init__(self, validset, topThresh):
        self.validset = validset
        self.topThresh = topThresh
        
    def reset(self):
        self.dices = {} 
        self.sens = {} 
        self.specs = {} 
        self.accs = {} 
        self.scores = {}
        
    def processOutputsToMask(self, outputs):
        scores = outputs['instances'].scores
        if len(scores)==0: 
            return torch.zeros(outputs['instances'].image_size).unsqueeze(0).cuda()
        else:
            good_indices = [i for i,x in enumerate(outputs['instances'].scores) if x>self.topThresh]
            if not good_indices: good_indices= [torch.argmax(scores).item()]
            #good_indices = torch.tensor(good_indices, dtype=torch.long).cuda()
        #print(good_indices)
        #print(outputs['instances'].pred_masks.size())
        mask = outputs['instances'].pred_masks.clone().detach()[good_indices]
        #mask = torch.index_select(outputs['instances'].pred_masks.clone().detach(), 0, good_indices)
        mask = mask.int().sum(dim=0).unsqueeze(0).cuda()
        mask = (mask>0).float()
        return mask
    
    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            id = input['image_id']
            pred_masks = pred_masks = self.processOutputsToMask(output)
            labels = torch.tensor(annsToSingleBinMask(self.validset, id)).cuda().unsqueeze(0)
            #labels = torch.tensor(self.validset.annToMask(self.validset.anns[id])).cuda().unsqueeze(0)
            #print(pred_masks.size(), labels.size())
            self.dices[id] = lossdice(pred_masks, labels).cpu().item()
            #print(Sens(pred_masks, labels).cpu().size())
            self.sens[id] = Sens(pred_masks, labels).cpu().item()
            self.specs[id] = Spec(pred_masks, labels).cpu().item()
            self.accs[id] = Acc(pred_masks, labels).cpu().item()
            
            if len(output['instances'].scores) == 0: scores = None
            elif len(output['instances'].scores) > 0: scores = list(output['instances'].scores.cpu().numpy()) 
    def evaluate(self):
        # save self.count somewhere, or print it, or return it.
        return {"dices": self.dices,
                "accs": self.accs,
                "sens": self.sens,
                "specs": self.specs,
                "scores": self.scores}


def save_results(results, path):
    with open(path, 'w') as file:
        json.dump(dict(results), file)

def mapper(dataset_dict):
	# Implement a mapper, similar to the default DatasetMapper, but with your own customizations
	dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
	image = utils.read_image(dataset_dict["file_name"], format="BGR")
	image, transforms = T.apply_transform_gens([T.Resize((256, 256))], image)
	dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

	annos = [
		utils.transform_instance_annotations(obj, transforms, image.shape[:2])
		for obj in dataset_dict.pop("annotations")
		if obj.get("iscrowd", 0) == 0
	]
	instances = utils.annotations_to_instances(annos, image.shape[:2])
	dataset_dict["instances"] = utils.filter_empty_instances(instances)
	return dataset_dict
