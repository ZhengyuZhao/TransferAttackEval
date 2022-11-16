import numpy as np
import cv2
import os
import pdb
import pickle
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import random

import matplotlib.pyplot as plt
import scipy.io as si
import shutil
from utils_data import *

##define TI
def gkern(kernlen=5, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def TI(grad_in, kernel_size=5):
    kernel = gkern(kernel_size, 3).astype(np.float32)
    gaussian_kernel = np.stack([kernel, kernel, kernel])
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
    gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()    
    grad_out = F.conv2d(grad_in, gaussian_kernel, bias=None, stride=1, padding=((kernel_size-1)/2,(kernel_size-1)/2), groups=3) #TI
    return grad_out


def TI_multi(X_in, in_size, kernel_size):
    a = (kernel_size-1)/2/in_size
    X_out = transforms.RandomAffine(0,translate=(a,a))(X_in)
    return X_out

##define DI
def DI(X_in, in_size, out_size):
    rnd = np.random.randint(in_size, out_size,size=1)[0]
    h_rem = out_size - rnd
    w_rem = out_size - rnd
    pad_top = np.random.randint(0, h_rem,size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem,size=1)[0]
    pad_right = w_rem - pad_left

    c = np.random.rand(1)
    if c <= 0.7:
        X_out = F.pad(F.interpolate(X_in, size=(rnd,rnd)), (pad_left,pad_top,pad_right,pad_bottom), mode='constant', value=0)
        return  X_out 
    else:
        return  X_in
    
    

#resnet50 inception_v3 densenet121 vgg16_bn
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

model_tar_1 = models.resnet50(pretrained=True)
model = models.inception_v3(pretrained=True,transform_input=True).eval()
model_tar_2 = models.densenet121(pretrained=True).eval()
model_tar_3 = models.vgg19_bn(pretrained=True).eval()


preprocess_layer = Preprocessing_Layer(mean,std)

model = nn.Sequential(preprocess_layer, model).eval()
model_tar_1 = nn.Sequential(preprocess_layer, model_tar_1).eval()
model_tar_2 = nn.Sequential(preprocess_layer, model_tar_2).eval()
model_tar_3 = nn.Sequential(preprocess_layer, model_tar_3).eval()

# model.eval()
model.to(device)
model_tar_1.to(device)
model_tar_2.to(device)
model_tar_3.to(device)

transform_299 = transforms.Compose([
transforms.Resize(299),
transforms.CenterCrop(299),
transforms.ToTensor(),
])


transform_224 = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
])   
 
# image_size = 299 for inception_v3 and image_size = 224 for resnet50, densenet121, and vgg16_bn  
image_size = 299
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]    
if image_size ==299:
    transform = transform_299
else:
    transform = transform_224

    
val_json = '../TransF5000_val.json'
val_loader = torch.utils.data.DataLoader(ImagePathDataset.from_path(config_path = val_json,transform=transform,return_paths=True),batch_size=40, shuffle=True,num_workers=1, pin_memory=True)

epsilon = 16.0 / 255.0
step_size = 2.0 / 255.0
num_iteration = 100
check_point = 5
multi_copies = 5


# DI
save_dir = os.path.join('../out','DI','incv3')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
pos = np.zeros((3,num_iteration // check_point))
    
for i, ((images, labels), path) in enumerate(val_loader):
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone()
    for j in range(num_iteration):
        img_x = img + img.new_zeros(img.size())
        img_x.requires_grad_(True)
        if not multi_copies:
            logits = model(DI(img_x,299,330))
            loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
            loss.backward()
            input_grad = img_x.grad.clone()
        else:               
            input_grad = 0
            for c in range(multi_copies):
                logits = model(DI(img_x,299,330))
                loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
                loss.backward()
                input_grad = input_grad + img_x.grad.clone()     
        img_x.grad.zero_()
        img = img.data + step_size * torch.sign(input_grad)
        img = torch.where(img > images + epsilon, images + epsilon, img)
        img = torch.where(img < images - epsilon, images - epsilon, img)
        img = torch.clamp(img, min=0, max=1)

        flag = (j+1) % check_point
        if flag == 0:
            point = j // check_point
            pos[0,point] = pos[0,point] + sum(torch.argmax(model_tar_1(img),dim=1) != labels).cpu().numpy()
            pos[1,point] = pos[1,point] + sum(torch.argmax(model_tar_2(img),dim=1) != labels).cpu().numpy()
            pos[2,point] = pos[2,point] + sum(torch.argmax(model_tar_3(img),dim=1) != labels).cpu().numpy()
        if j == 49: 
            save_images(img.detach().cpu().numpy(), img_list=path, idx=len(path), output_dir=save_dir)
print(pos)


# TI
save_dir = os.path.join('../out','TI','incv3')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
pos = np.zeros((3,num_iteration // check_point))
pos_resize = np.zeros((3,num_iteration // check_point))
  
for i, ((images, labels), path) in enumerate(val_loader):
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone()
    for j in range(num_iteration):
        img_x = img + img.new_zeros(img.size())
        img_x.requires_grad_(True)
        if not multi_copies:
            logits = model(TI(img_x))
            loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
            loss.backward()
            input_grad = img_x.grad.clone()
            input_grad = TI(input_grad)
        else:               
            input_grad = 0
            for c in range(multi_copies):
                logits = model(TI_multi(img_x,image_size,5))
#                 TI_multi(X_in, in_size, kernel_size)
                loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
                loss.backward()
                input_grad = input_grad + img_x.grad.clone()     
        img_x.grad.zero_()
        img = img.data + step_size * torch.sign(input_grad)
        img = torch.where(img > images + epsilon, images + epsilon, img)
        img = torch.where(img < images - epsilon, images - epsilon, img)
        img = torch.clamp(img, min=0, max=1)

        flag = (j+1) % check_point
        if flag == 0:
            point = j // check_point
            pos[0,point] = pos[0,point] + sum(torch.argmax(model_tar_1(img),dim=1) != labels).cpu().numpy()
            pos[1,point] = pos[1,point] + sum(torch.argmax(model_tar_2(img),dim=1) != labels).cpu().numpy()
            pos[2,point] = pos[2,point] + sum(torch.argmax(model_tar_3(img),dim=1) != labels).cpu().numpy()                        
        if j == 49: 
            save_images(img.detach().cpu().numpy(), img_list=path, idx=len(path), output_dir=save_dir)              
print(pos)


# SI
save_dir = os.path.join('../out','SI','incv3')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
pos = np.zeros((3,num_iteration // check_point))
    
for i, ((images, labels), path) in enumerate(val_loader):
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone()
    for j in range(num_iteration):
        img_x = img + img.new_zeros(img.size())
        img_x.requires_grad_(True)
        if not multi_copies:
            logits = model(img_x * random.randint(5, 10)/10)
            loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
            loss.backward()
            input_grad = img_x.grad.clone()
        else:               
            input_grad = 0
#             for i in range(multi_copies):
            for c in [1,2,4,8,16]:
                logits = model(img_x / c)
                loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
                loss.backward()
                input_grad = input_grad + img_x.grad.clone()     
        img_x.grad.zero_()
        img = img.data + step_size * torch.sign(input_grad)
        img = torch.where(img > images + epsilon, images + epsilon, img)
        img = torch.where(img < images - epsilon, images - epsilon, img)
        img = torch.clamp(img, min=0, max=1)

        flag = (j+1) % check_point
        if flag == 0:
            point = j // check_point
            pos[0,point] = pos[0,point] + sum(torch.argmax(model_tar_1(img),dim=1) != labels).cpu().numpy()
            pos[1,point] = pos[1,point] + sum(torch.argmax(model_tar_2(img),dim=1) != labels).cpu().numpy()
            pos[2,point] = pos[2,point] + sum(torch.argmax(model_tar_3(img),dim=1) != labels).cpu().numpy()
        if j == 49: 
            save_images(img.detach().cpu().numpy(), img_list=path, idx=len(path), output_dir=save_dir)
print(pos)


# VT
save_dir = os.path.join('..','VT','incv3')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
pos = np.zeros((3,num_iteration // check_point)) 
for i, ((images, labels), path) in enumerate(val_loader):
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone()
    variance = 0
    for j in range(num_iteration):
        img_x = img
        img_x.requires_grad_(True)
        logits = model(img_x)
        loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
        loss.backward()
        new_grad = img_x.grad.clone()    
        
        global_grad = 0
        for c in range(multi_copies):
            logits = model(img_x + img.new(img.size()).uniform_(-1.5 * epsilon,1.5 * epsilon))
            loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
            loss.backward()
            global_grad = global_grad + img_x.grad.clone()  
            
        input_grad = new_grad + variance 
        variance = global_grad / multi_copies - new_grad
        img_x.grad.zero_()
        img = img.data + step_size * torch.sign(input_grad)
        img = torch.where(img > images + epsilon, images + epsilon, img)
        img = torch.where(img < images - epsilon, images - epsilon, img)
        img = torch.clamp(img, min=0, max=1)
    
        flag = (j+1) % check_point
        if flag == 0:
            point = j // check_point
            pos[0,point] = pos[0,point] + sum(torch.argmax(model_tar_1(img),dim=1) != labels).cpu().numpy()
            pos[1,point] = pos[1,point] + sum(torch.argmax(model_tar_2(img),dim=1) != labels).cpu().numpy()
            pos[2,point] = pos[2,point] + sum(torch.argmax(model_tar_3(img),dim=1) != labels).cpu().numpy()
        if j == 49: 
            save_images(img.detach().cpu().numpy(), img_list=path, idx=len(path), output_dir=save_dir)             
print(pos)


# Admix
save_dir = os.path.join('..','Admix','incv3')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
pos = np.zeros((3,num_iteration // check_point))
for i, ((images, labels), path) in enumerate(val_loader):
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone()
    for j in range(num_iteration):
        img_x = img
        img_x.requires_grad_(True) 
        input_grad = 0
        for c in range(multi_copies):
            img_other = img[torch.randperm(img.shape[0])].view(img.size())
            logits = model(img_x + 0.2 * img_other)
            loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
            loss.backward()
            input_grad = input_grad + img_x.grad.clone()  
            
        img_x.grad.zero_()
        img = img.data + step_size * torch.sign(input_grad)
        img = torch.where(img > images + epsilon, images + epsilon, img)
        img = torch.where(img < images - epsilon, images - epsilon, img)
        img = torch.clamp(img, min=0, max=1)

        flag = (j+1) % check_point
        if flag == 0:
            point = j // check_point
            pos[0,point] = pos[0,point] + sum(torch.argmax(model_tar_1(img),dim=1) != labels).cpu().numpy()
            pos[1,point] = pos[1,point] + sum(torch.argmax(model_tar_2(img),dim=1) != labels).cpu().numpy()
            pos[2,point] = pos[2,point] + sum(torch.argmax(model_tar_3(img),dim=1) != labels).cpu().numpy()

        if j == 49: 
            save_images(img.detach().cpu().numpy(), img_list=path, idx=len(path), output_dir=save_dir)                       
print(pos)
