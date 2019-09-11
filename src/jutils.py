#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 23:09:46 2019

@author: arjun
"""

#jeremies stuff
from functools import partial
import torch, math, os
import json
import numpy as np

def annealer(f):
    def _inner(start, end): return partial(f, start, end)
    return _inner

@annealer
def sched_lin(start, end, pos): return start + pos*(end-start)

@annealer
def sched_cos(start, end, pos): return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2

@annealer
def sched_no(start, end, pos):  return start

@annealer
def sched_exp(start, end, pos): return start * (end/start) ** pos

#This monkey-patch is there to be able to plot tensors
torch.Tensor.ndim = property(lambda x: len(x.shape))

def combine_scheds(pcts, scheds):
    assert sum(pcts) == 1.
    pcts = torch.tensor([0] + listify(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)
    def _inner(pos):
        idx = (pos >= pcts).nonzero().max()
        actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])
        return scheds[idx](actual_pos)
    return _inner


def cos_1cycle_anneal(start, high, end):
    return [sched_cos(start, high), sched_cos(high, end)]


def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]

def jsonsaveddict(dictionary, save_dir, name):
    #dictionary is a dict, name is a string 
    aaa = json.dumps(dictionary)
    f = open(os.path.join(save_dir, name),"w")
    f.write(aaa)
    f.close()
   
def scalar_thresh(cappy, threshes):
    _cappy = torch.tensor(cappy)[0]#get rid of batch size
    threshes = torch.tensor(threshes)[:,None,None] #add xydim
    binary = torch.tensor(torch.gt(_cappy, threshes), dtype=torch.float)
    return binary    


def sens(c, l):
    inter = torch.sum(c*l, dim=(1,2))
    union = torch.sum(c, dim=(1,2)) + torch.sum(l) - inter
    return inter/union

def spec(c, l):
    c=1-c
    l=1-l
    inter = torch.sum(c*l, dim=(1,2))
    union = torch.sum(c, dim=(1,2)) + torch.sum(l) - inter
    return inter/union


def acc(c, l):
    return torch.sum(c,dim=(1,2)) / torch.prod(torch.tensor(l.size()[-2:-1]))

def jsonloaddict(analy_path, dict_name):
    f = open(os.path.join(analy_path, dict_name + '.json'))
    dictionary = json.load(f)
    f.close()
    return dictionary

def dict_to_difficulty(dictionary, threshold):
    for k,v in dictionary.items():
        if v<0.96:
            dictionary[k] = 1 # 9.3656 is class imbalance coefficient
        else:
            dictionary[k] = 0
    return dictionary

def get_hard(dictionary):
    hardnames=[]
    for k,v in dictionary.items():
        if v!=0:
            hardnames.append(k)
    return hardnames

def get_classifier_results(valloss, hardnames):
    answers=list(valloss.values())
    total_percent_correct = np.array(answers).sum()/len(answers)
    for k,v in valloss.items():
        hardcorrect = [True for a in hardnames if a==k and v==True]
    hardcorrect = np.array(hardcorrect).sum()/len(hardnames)
    return 100*total_percent_correct, 100*hardcorrect



'''
def sens(c,l):
    intersection = torch.sum(c * l)
    union = torch.sum(c) + torch.sum(l) - intersection
    loss = (intersection) / (union)
    return loss

def spec(c,l):
    c=1-c
    l=1-l
    intersection = torch.sum(c * l)
    union = torch.sum(c) + torch.sum(l) - intersection
    loss = (intersection) / (union)
    return loss


def scalar_thresh(cappy, thresh):
    _cappy = torch.tensor(cappy)
    _cappy[_cappy>=thresh] = 1.
    _cappy[_cappy<thresh] = 0.
    return _cappy
    
'''