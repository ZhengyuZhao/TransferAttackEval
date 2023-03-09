import numpy as np
import cv2
import os
import pdb
import pickle
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import torch.utils.data as Data
import torch.nn.functional as F
import dill

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import random

import matplotlib.pyplot as plt
import scipy.io as si
import shutil
from utils_data import *

from typing import Type, Any, Callable, Union, List, Optional

## SGM utils
def backward_hook(gamma):
    # implement SGM through grad through ReLU
    def _backward_hook(module, grad_in, grad_out):
        if isinstance(module, nn.ReLU):
            return (gamma * grad_in[0],)
    return _backward_hook


def backward_hook_norm(module, grad_in, grad_out):
    # normalize the gradient to avoid gradient explosion or vanish
    std = torch.std(grad_in[0])
    return (grad_in[0] / std,)


def register_hook_for_resnet(model, arch, gamma):
    # There is only 1 ReLU in Conv module of ResNet-18/34
    # and 2 ReLU in Conv module ResNet-50/101/152
    if arch in ['resnet50', 'resnet101', 'resnet152']:
        gamma = np.power(gamma, 0.5)
    backward_hook_sgm = backward_hook(gamma)

    for name, module in model.named_modules():
        if 'relu' in name and not '0.relu' in name:
            module.register_backward_hook(backward_hook_sgm)

        # e.g., 1.layer1.1, 1.layer4.2, ...
        # if len(name.split('.')) == 3:
        if len(name.split('.')) >= 2 and 'layer' in name.split('.')[-2]:
            module.register_backward_hook(backward_hook_norm)


def register_hook_for_densenet(model, arch, gamma):
    # There are 2 ReLU in Conv module of DenseNet-121/169/201.
    gamma = np.power(gamma, 0.5)
    backward_hook_sgm = backward_hook(gamma)
    for name, module in model.named_modules():
        if 'relu' in name and not 'transition' in name:
            module.register_backward_hook(backward_hook_sgm)

## LinBP utils
def linbp_forw_resnet50(model, x, do_linbp, linbp_layer):
    jj = int(linbp_layer.split('_')[0])
    kk = int(linbp_layer.split('_')[1])
    x = model[0](x)
    x = model[1].conv1(x)
    x = model[1].bn1(x)
    x = model[1].relu(x)
    x = model[1].maxpool(x)
    ori_mask_ls = []
    conv_out_ls = []
    relu_out_ls = []
    conv_input_ls = []
    def layer_forw(jj, kk, jj_now, kk_now, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp):
        if jj < jj_now:
            x, ori_mask, conv_out, relu_out, conv_in = block_func(mm, x, linbp=True)
            ori_mask_ls.append(ori_mask)
            conv_out_ls.append(conv_out)
            relu_out_ls.append(relu_out)
            conv_input_ls.append(conv_in)
        elif jj == jj_now:
            if kk_now >= kk:
                x, ori_mask, conv_out, relu_out, conv_in = block_func(mm, x, linbp=True)
                ori_mask_ls.append(ori_mask)
                conv_out_ls.append(conv_out)
                relu_out_ls.append(relu_out)
                conv_input_ls.append(conv_in)
            else:
                x, _, _, _, _ = block_func(mm, x, linbp=False)
        else:
            x, _, _, _, _ = block_func(mm, x, linbp=False)
        return x, ori_mask_ls
    for ind, mm in enumerate(model[1].layer1):
        x, ori_mask_ls = layer_forw(jj, kk, 1, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp)
    for ind, mm in enumerate(model[1].layer2):
        x, ori_mask_ls = layer_forw(jj, kk, 2, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp)
    for ind, mm in enumerate(model[1].layer3):
        x, ori_mask_ls = layer_forw(jj, kk, 3, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp)
    for ind, mm in enumerate(model[1].layer4):
        x, ori_mask_ls = layer_forw(jj, kk, 4, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp)
    x = model[1].avgpool(x)
    x = torch.flatten(x, 1)
    x = model[1].fc(x)
    return x, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls

def block_func(block, x, linbp):
    identity = x
    conv_in = x+0
    out = block.conv1(conv_in)
    out = block.bn1(out)
    out_0 = out + 0
    if linbp:
        out = linbp_relu(out_0)
    else:
        out = block.relu(out_0)
    ori_mask_0 = out.data.bool().int()

    out = block.conv2(out)
    out = block.bn2(out)
    out_1 = out + 0
    if linbp:
        out = linbp_relu(out_1)
    else:
        out = block.relu(out_1)
    ori_mask_1 = out.data.bool().int()

    out = block.conv3(out)
    out = block.bn3(out)

    if block.downsample is not None:
        identity = block.downsample(identity)
    identity_out = identity + 0
    x_out = out + 0


    out = identity_out + x_out
    out = block.relu(out)
    ori_mask_2 = out.data.bool().int()
    return out, (ori_mask_0, ori_mask_1, ori_mask_2), (identity_out, x_out), (out_0, out_1), (0, conv_in)

def linbp_relu(x):
    x_p = F.relu(-x)
    x = x + x_p.data
    return x

def linbp_backw_resnet50(img, loss, conv_out_ls, ori_mask_ls, relu_out_ls, conv_input_ls, xp):
    for i in range(-1, -len(conv_out_ls)-1, -1):
        if i == -1:
            grads = torch.autograd.grad(loss, conv_out_ls[i])
        else:
            grads = torch.autograd.grad((conv_out_ls[i+1][0], conv_input_ls[i+1][1]), conv_out_ls[i], grad_outputs=(grads[0], main_grad_norm))
        normal_grad_2 = torch.autograd.grad(conv_out_ls[i][1], relu_out_ls[i][1], grads[1]*ori_mask_ls[i][2],retain_graph=True)[0]
        normal_grad_1 = torch.autograd.grad(relu_out_ls[i][1], relu_out_ls[i][0], normal_grad_2 * ori_mask_ls[i][1], retain_graph=True)[0]
        normal_grad_0 = torch.autograd.grad(relu_out_ls[i][0], conv_input_ls[i][1], normal_grad_1 * ori_mask_ls[i][0], retain_graph=True)[0]
        del normal_grad_2, normal_grad_1
        main_grad = torch.autograd.grad(conv_out_ls[i][1], conv_input_ls[i][1], grads[1])[0]
        alpha = normal_grad_0.norm(p=2, dim = (1,2,3), keepdim = True) / main_grad.norm(p=2,dim = (1,2,3), keepdim=True)
        main_grad_norm = xp * alpha * main_grad
    input_grad = torch.autograd.grad((conv_out_ls[0][0], conv_input_ls[0][1]), img, grad_outputs=(grads[0], main_grad_norm))
    return input_grad[0].data

## RFA load pre-trained model

## IAA defined in utils_iaa.py

## DSM load pre-trained model

## load imagenet
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
 
image_size = 224
if image_size ==299:
    transform = transform_299
else:
    transform = transform_224

val_json = '../TransF5000_val.json'

val_loader = torch.utils.data.DataLoader(ImagePathDataset.from_path(config_path = val_json,transform=transform,return_paths=True),batch_size=40, shuffle=True,num_workers=1, pin_memory=True)

### parameters
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
preprocess_layer = Preprocessing_Layer(mean,std)

epsilon = 16.0 / 255.0
step_size = 2.0 / 255.0
num_iteration = 100
check_point = 5

### evaluation models
model_tar_1 = models.inception_v3(pretrained=True,transform_input=True).eval()
model_tar_2 = models.densenet121(pretrained=True).eval()
model_tar_3 = models.vgg19_bn(pretrained=True).eval()

model_tar_1 = nn.Sequential(preprocess_layer, model_tar_1).eval()
model_tar_2 = nn.Sequential(preprocess_layer, model_tar_2).eval()
model_tar_3 = nn.Sequential(preprocess_layer, model_tar_3).eval()

model_tar_1.to(device)
model_tar_2.to(device)
model_tar_3.to(device)


## SGM
save_dir = os.path.join('../out','SGM','res50')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
pos = np.zeros((3,num_iteration // check_point))
resnet = models.resnet50()
resnet = nn.Sequential(preprocess_layer, resnet)
resnet.to(device)
resnet.eval()
gamma = 0.5
register_hook_for_resnet(resnet, arch='resnet50', gamma=gamma)

for i, ((images, labels), path) in enumerate(val_loader):
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone()
    for j in range(num_iteration):
        img_x = img
        img_x.requires_grad_(True)
        att_out = resnet(img_x)
        pred = torch.argmax(att_out, dim=1).view(-1)
        loss = nn.CrossEntropyLoss()(att_out, labels)
        resnet.zero_grad()
        loss.backward()
        input_grad = img_x.grad.data
        resnet.zero_grad()
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


## LinBP
save_dir = os.path.join('../out','LinBP','res50')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
pos = np.zeros((3,num_iteration // check_point))
resnet = models.resnet50()
resnet = nn.Sequential(preprocess_layer, resnet)
resnet.to(device)
resnet.eval()
linbp_layer = '3_1'
sgm_lambda = 1.0

for i, ((images, labels), path) in enumerate(val_loader):
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone()
    for j in range(num_iteration):
        img_x = img
        img_x.requires_grad_(True)
        att_out, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls = linbp_forw_resnet50(resnet, img_x, True, linbp_layer)
        pred = torch.argmax(att_out, dim=1).view(-1)
        loss = nn.CrossEntropyLoss()(att_out, labels)
        resnet.zero_grad()
        input_grad = linbp_backw_resnet50(img_x, loss, conv_out_ls, ori_mask_ls, relu_out_ls, conv_input_ls, xp=sgm_lambda)
        resnet.zero_grad()
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


## RFA 
## We follow the original work of RFA to use the pre-trained models provided in: https://github.com/microsoft/robust-models-transfer. 
## Specifically, we report results for L2-Robust ImageNet Model with ResNet50 and ε=0.1 as well as Linf-Robust ImageNet Model with ResNet50 and ε=8.
save_dir = os.path.join('../out','RFA','res50')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
pos = np.zeros((3,num_iteration // check_point))
resnet = models.resnet50()
model_path = '../models/imagenet_linf_8.pt'
state_dict = torch.load(model_path, pickle_module=dill)
sd = state_dict['model']
for key in list(sd.keys()):
    if 'attacker.' in key:
        del sd[key]
    elif 'module.model.' in key:
        sd[key.replace('module.model.','')] = sd[key]
        del sd[key]
    elif 'module.normalizer.' in key:
        del sd[key]
model_dict = resnet.load_state_dict(sd)
resnet = nn.Sequential(preprocess_layer, resnet)
resnet.to(device)
resnet.eval()

for i, ((images, labels), path) in enumerate(val_loader):
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone()
    for j in range(num_iteration):
        img_x = img
        img_x.requires_grad_(True)
        att_out = resnet(img_x)
        pred = torch.argmax(att_out, dim=1).view(-1)
        loss = nn.CrossEntropyLoss()(att_out, labels)
        resnet.zero_grad()
        loss.backward()
        input_grad = img_x.grad.data
        resnet.zero_grad()
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


## IAA
from utils_iaa import resnet50
save_dir = os.path.join('../out','IAA','res50')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
pos = np.zeros((3,num_iteration // check_point))
resnet = resnet50()
model_path = '../models/resnet50-0676ba61.pth'
pre_dict = torch.load(model_path)
resnet_dict = resnet.state_dict()
state_dict = {k:v for k,v in pre_dict.items() if k in resnet_dict.keys()}
print("loaded pretrained weight. Len:",len(pre_dict.keys()),len(state_dict.keys()))
resnet_dict.update(state_dict)
model_dict = resnet.load_state_dict(resnet_dict)
resnet = nn.Sequential(preprocess_layer, resnet)
resnet.to(device)
resnet.eval()

for i, ((images, labels), path) in enumerate(val_loader):
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone()
    for j in range(num_iteration):
        img_x = img
        img_x.requires_grad_(True)
        att_out = resnet(img_x)
        pred = torch.argmax(att_out, dim=1).view(-1)
        loss = nn.CrossEntropyLoss()(att_out, labels)
        resnet.zero_grad()
        loss.backward()
        input_grad = img_x.grad.data
        resnet.zero_grad()
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


## DSM
save_dir = os.path.join('../out','DSM','res50')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
pos = np.zeros((3,num_iteration // check_point))
resnet = models.resnet18()
model_path = '../models/SD_resnet18_cutmix.pth.tar'
state_dict = torch.load(model_path)
sd = state_dict['state_dict']
for key in list(sd.keys()):
    sd[key.replace('module.','')] = sd[key]
    del sd[key]
model_dict = resnet.load_state_dict(sd)
resnet = nn.Sequential(preprocess_layer, resnet)
resnet.to(device)
resnet.eval()

for i, ((images, labels), path) in enumerate(val_loader):
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone()
    for j in range(num_iteration):
        img_x = img
        img_x.requires_grad_(True)
        att_out = resnet(img_x)
        pred = torch.argmax(att_out, dim=1).view(-1)
        loss = nn.CrossEntropyLoss()(att_out, labels)
        resnet.zero_grad()
        loss.backward()
        input_grad = img_x.grad.data
        resnet.zero_grad()
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
