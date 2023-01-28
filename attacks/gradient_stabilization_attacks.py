import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.utils
from torchvision import models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights,Inception_V3_Weights,DenseNet121_Weights,VGG19_BN_Weights
from utils_data import *

#resnet50 inception_v3 densenet121 vgg16_bn
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

model_tar_1 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,transform_input=True).eval()
model_tar_2 = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).eval()
model_tar_3 = models.vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1).eval()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]   
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
 
if image_size ==299:
    transform = transform_299
else:
    transform = transform_224

    
val_json = '../TransF5000_val.json'
val_loader = torch.utils.data.DataLoader(ImagePathDataset.from_path(config_path = val_json,transform=transform,return_paths=True),batch_size=50, shuffle=False,num_workers=1, pin_memory=True)

epsilon = 16.0 / 255.0
step_size = 2.0 / 255.0
num_iteration = 100
check_point = 5


# PGD (baseline)
save_dir = os.path.join('../out','PGD','incv3')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
suc = np.zeros((3,num_iteration // check_point))

# for i in range(1):
#     ((images, labels), path)= next(iter(val_loader))
for i, ((images, labels), path) in enumerate(val_loader):
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone()
    for j in range(num_iteration):
        img_x = img + img.new_zeros(img.size())
        img_x.requires_grad_(True)
        logits = model(img_x)
        loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
        loss.backward()
        input_grad = img_x.grad.clone()
        img_x.grad.zero_()
        img = img.data + step_size * torch.sign(input_grad)
        img = torch.where(img > images + epsilon, images + epsilon, img)
        img = torch.where(img < images - epsilon, images - epsilon, img)
        img = torch.clamp(img, min=0, max=1)

        flag = (j+1) % check_point
        if flag == 0:
            point = j // check_point
            suc[0,point] = suc[0,point] + sum(torch.argmax(model_tar_1(img),dim=1) != labels).cpu().numpy()
            suc[1,point] = suc[1,point] + sum(torch.argmax(model_tar_2(img),dim=1) != labels).cpu().numpy()
            suc[2,point] = suc[2,point] + sum(torch.argmax(model_tar_3(img),dim=1) != labels).cpu().numpy()
        if j == 49: 
            save_images(img.detach().cpu().numpy(), img_list=path, idx=len(path), output_dir=save_dir)
print('PGD success rate:')
print(suc/5000)


# MI
save_dir = os.path.join('../out','MI','incv3')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
suc = np.zeros((3,num_iteration // check_point))
     
for i, ((images, labels), path) in enumerate(val_loader):
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone()
    grad_pre = 0
    for j in range(num_iteration):
        img_x = img + img.new_zeros(img.size())
        img_x.requires_grad_(True)
        logits = model(img_x)
        loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
        loss.backward()
        input_grad = img_x.grad.clone()
        input_grad = input_grad / torch.mean(torch.abs(input_grad), (1, 2, 3), keepdim=True) + 1 * grad_pre
        grad_pre = input_grad 

        img_x.grad.zero_()
        img = img.data + step_size * torch.sign(input_grad)
        img = torch.where(img > images + epsilon, images + epsilon, img)
        img = torch.where(img < images - epsilon, images - epsilon, img)
        img = torch.clamp(img, min=0, max=1)

        flag = (j+1) % check_point
        if flag == 0:
            point = j // check_point
            suc[0,point] = suc[0,point] + sum(torch.argmax(model_tar_1(img),dim=1) != labels).cpu().numpy()
            suc[1,point] = suc[1,point] + sum(torch.argmax(model_tar_2(img),dim=1) != labels).cpu().numpy()
            suc[2,point] = suc[2,point] + sum(torch.argmax(model_tar_3(img),dim=1) != labels).cpu().numpy()
        if j == 9: 
            save_images(img.detach().cpu().numpy(), img_list=path, idx=len(path), output_dir=save_dir)
print('MI success rate:')
print(suc/5000)


# NI
save_dir = os.path.join('../out','NI','incv3')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
suc = np.zeros((3,num_iteration // check_point))

for i, ((images, labels), path) in enumerate(val_loader):
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone()
    grad_pre = 0
    for j in range(num_iteration):
        img_x = img + step_size * 1 * grad_pre
        img_x.requires_grad_(True)
        logits = model(img_x)
        loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
        loss.backward()
        input_grad = img_x.grad.clone()
        input_grad = input_grad / torch.mean(torch.abs(input_grad), (1, 2, 3), keepdim=True) + 1 * grad_pre #MI
        grad_pre = input_grad 
        img_x.grad.zero_()
        img = img.data + step_size * torch.sign(input_grad)
        img = torch.where(img > images + epsilon, images + epsilon, img)
        img = torch.where(img < images - epsilon, images - epsilon, img)
        img = torch.clamp(img, min=0, max=1)

        flag = (j+1) % check_point
        if flag == 0:
            point = j // check_point
            suc[0,point] = suc[0,point] + sum(torch.argmax(model_tar_1(img),dim=1) != labels).cpu().numpy()
            suc[1,point] = suc[1,point] + sum(torch.argmax(model_tar_2(img),dim=1) != labels).cpu().numpy()
            suc[2,point] = suc[2,point] + sum(torch.argmax(model_tar_3(img),dim=1) != labels).cpu().numpy()
            
        if j == 9: 
            save_images(img.detach().cpu().numpy(), img_list=path, idx=len(path), output_dir=save_dir)
print('NI success rate:')
print(suc/5000)


# PI
save_dir = os.path.join('../out','PI','incv3')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
suc = np.zeros((3,num_iteration // check_point))

for i, ((images, labels), path) in enumerate(val_loader):
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone()
    grad_pre = 0
    grad_t = 0
    for j in range(num_iteration):
        img_x = img + step_size * 1 * grad_t
        img_x.requires_grad_(True)
        logits = model(img_x)
        loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
        loss.backward()
        grad_t = img_x.grad.clone()
        grad_t = grad_t / torch.mean(torch.abs(grad_t), (1, 2, 3), keepdim=True)
        input_grad = grad_t + 1 * grad_pre #MI
        grad_pre = input_grad 
        img_x.grad.zero_()
        img = img.data + step_size * torch.sign(input_grad)
        img = torch.where(img > images + epsilon, images + epsilon, img)
        img = torch.where(img < images - epsilon, images - epsilon, img)
        img = torch.clamp(img, min=0, max=1)

        flag = (j+1) % check_point
        if flag == 0:
            point = j // check_point
            suc[0,point] = suc[0,point] + sum(torch.argmax(model_tar_1(img),dim=1) != labels).cpu().numpy()
            suc[1,point] = suc[1,point] + sum(torch.argmax(model_tar_2(img),dim=1) != labels).cpu().numpy()
            suc[2,point] = suc[2,point] + sum(torch.argmax(model_tar_3(img),dim=1) != labels).cpu().numpy()
        if j == 9: 
            save_images(img.detach().cpu().numpy(), img_list=path, idx=len(path), output_dir=save_dir)       
print('PI success rate:')
print(suc/5000)
