#!/usr/bin/env python3
# iFGSM, TAP and ILA inherited and modified from the ILA repository (https://github.com/CUAI/Intermediate-Level-Attack/blob/master/attacks.py)
import random
import os
random.seed(0)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import pickle
from tqdm import tqdm


# helpers
def avg(l):
    return sum(l) / len(l)

class NormalizeInverse(torchvision.transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean).cuda()
        std = torch.as_tensor(std).cuda()
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

def sow_images(images):
    """Sow batch of torch images (Bx3xWxH) into a grid PIL image (BWxHx3)

    Args:
        images: batch of torch images.

    Returns:
        The grid of images, as a numpy array in PIL format.
    """
    images = torchvision.utils.make_grid(
        images
    )  # sow our batch of images e.g. (4x3x32x32) into a grid e.g. (3x32x128)
    
    mean_arr, stddev_arr = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    # denormalize
    for c in range(3):
        images[c, :, :] = images[c, :, :] * stddev_arr[c] + mean_arr[c]

    images = images.cpu().numpy()  # go from Tensor to numpy array
    # switch channel order back from
    # torch Tensor to PIL image: going from 3x32x128 - to 32x128x3
    images = np.transpose(images, (1, 2, 0))
    return images

invNormalize = NormalizeInverse([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
Normlize_Trans = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
def update_adv(X, X_pert, pert, epsilon):
    X = X.clone().detach()
    X_pert = X_pert.clone().detach()
    X_pert = invNormalize(X_pert)
    X = invNormalize(X)
    X_pert = X_pert + pert
    noise = (X_pert - X).clamp(-epsilon, epsilon)
    X_pert = X + noise
    X_pert = X_pert.clamp(0, 1)
    X_pert = Normlize_Trans(X_pert)
    return X_pert.clone().detach()


def ifgsm(
    model,
    X,
    y,
    niters=10,
    epsilon=0.03,
    visualize=False,
    learning_rate=0.005,
    display=4,
    defense_model=False,
    setting="regular",
    dataset="cifar10",
    use_Inc_model = False,
):
    """Perform ifgsm attack with respect to model on images X with labels y

    Args:
        model: torch model with respect to which attacks will be computed
        X: batch of torch images
        y: labels corresponding to the batch of images
        niters: number of iterations of ifgsm to perform
        epsilon: Linf norm of resulting perturbation; scale of images is -1..1
        visualize: whether you want to visualize the perturbations or not
        learning_rate: learning rate of ifgsm
        display: number of images to display in visualization
        defense_model: set to true if you are using a defended model,
        e.g. ResNet18Defended, instead of the usual ResNet18
        setting: 'regular' is usual ifgsm, 'll' is least-likely ifgsm, and
        'nonleaking' is non-label-leaking ifgsm
        dataset: dataset the images are from, 'cifar10' | 'imagenet'

    Returns:
        The batch of adversarial examples corresponding to the original images
    """
    model.eval()
    out = None
    if defense_model:
        out = model(X)[0]
    else:
        out = model(X)
    y_ll = out.min(1)[1]  # least likely model output
    y_ml = out.max(1)[1]  # model label

    X_pert = X.clone()
    X_pert.requires_grad = True
    for i in range(niters):
        output_perturbed = None
        if defense_model:
            output_perturbed = model(X_pert)[0]
        else:
            output_perturbed = model(X_pert)

        y_used = y
        ll_factor = 1
        if setting == "ll":
            y_used = y_ll
            ll_factor = -1
        elif setting == "noleaking":
            y_used = y_ml

        loss = nn.CrossEntropyLoss()(output_perturbed, y_used)
        loss.backward()
        pert = ll_factor * learning_rate * X_pert.grad.detach().sign()

        # perform visualization
        if visualize is True and i == niters - 1:
            np_image = sow_images(X[:display].detach())
            np_delta = sow_images(pert[:display].detach())
            np_recons = sow_images(
                (X_pert.detach() + pert.detach()).clamp(-1, 1)[:display]
            )

            fig = plt.figure(figsize=(8, 8))
            fig.add_subplot(3, 1, 1)
            plt.axis("off")
            plt.imshow(np_recons)
            fig.add_subplot(3, 1, 2)
            plt.axis("off")
            plt.imshow(np_image)
            fig.add_subplot(3, 1, 3)
            plt.axis("off")
            plt.imshow(np_delta)
            plt.show()
        # end visualization

        X_pert = update_adv(X, X_pert, pert, epsilon)
        X_pert.requires_grad = True        

    return X_pert.clone().detach()


# TAP (transferable adversairal perturbation ECCV 2018)
class Transferable_Adversarial_Perturbations_Loss(torch.nn.Module):
    def __init__(self):
        super(Transferable_Adversarial_Perturbations_Loss, self).__init__()

    def forward(
        self,
        X,
        X_pert,
        original_mids,
        new_mids,
        y,
        output_perturbed,
        lam,
        alpha,
        s,
        yita,
    ):

        l1 = nn.CrossEntropyLoss()(output_perturbed, y)

        l2 = 0
        for i, new_mid in enumerate(new_mids):
            a = torch.sign(original_mids[i]) * torch.pow(
                torch.abs(original_mids[i]), alpha
            )
            b = torch.sign(new_mid) * torch.pow(torch.abs(new_mid), alpha)
            l2 += lam * (a - b).norm() ** 2

        l3 = yita * torch.abs(nn.AvgPool2d(s)(X - X_pert)).sum()

        return l1 + l2 + l3


mid_outputs = []
mid_grads = []

def Transferable_Adversarial_Perturbations(
    model,
    X,
    y,
    niters=10,
    epsilon=0.03,
    lam=0.005,
    alpha=0.5,
    s=3,
    yita=0.01,
    learning_rate=0.006,
    dataset="cifar10",
    use_Inc_model = False,
):
    """Perform cifar10 TAP attack using model on images X with labels y

    Args:
        model: torch model with respect to which attacks will be computed
        X: batch of torch images
        y: labels corresponding to the batch of images
        niters: number of iterations of TAP to perform
        epsilon: Linf norm of resulting perturbation; scale of images is -1..1
        lam: lambda parameter of TAP
        alpha: alpha parameter of TAP
        s: s parameter of TAP
        yita: yita parameter of TAP
        learning_rate: learning rate of TAP attack

    Returns:
        The batch of adversarial examples corresponding to the original images
    """
    feature_layers = list(model._modules.keys())
    global mid_outputs
    X = X.detach()
    X_pert = torch.zeros(X.size()).cuda()
    X_pert.copy_(X).detach()
    X_pert.requires_grad = True

    def get_mid_output(m, i, o):
        global mid_outputs
        mid_outputs.append(o)

    hs = []
    for layer_name in feature_layers:
        hs.append(model._modules.get(layer_name).register_forward_hook(get_mid_output))

    out = model(X)
        
    mid_originals = []
    for mid_output in mid_outputs:
        mid_original = torch.zeros(mid_output.size()).cuda()
        mid_originals.append(mid_original.copy_(mid_output))

    mid_outputs = []
    adversaries = []

    for iter_n in range(niters):
        output_perturbed = model(X_pert)
        # generate adversarial example by max middle
        # layer pertubation in the direction of increasing loss
        mid_originals_ = []
        for mid_original in mid_originals:
            mid_originals_.append(mid_original.detach())

        loss = Transferable_Adversarial_Perturbations_Loss()(
            X,
            X_pert,
            mid_originals_,
            mid_outputs,
            y,
            output_perturbed,
            lam,
            alpha,
            s,
            yita,
        )
        loss.backward()
        pert = learning_rate * X_pert.grad.detach().sign()

        X_pert = update_adv(X, X_pert, pert, epsilon)
        X_pert.requires_grad = True        

        mid_outputs = []

        if (iter_n+1) % 10 == 0:
            adversaries.append(X_pert.clone().detach())

    for h in hs:
        h.remove()
    return X_pert.clone().detach()
    # return adversaries


# ILA attack

# square sum of dot product
class Proj_Loss(torch.nn.Module):
    def __init__(self):
        super(Proj_Loss, self).__init__()

    def forward(self, old_attack_mid, new_mid, original_mid, coeff):
        x = (old_attack_mid - original_mid).view(1, -1)
        y = (new_mid - original_mid).view(1, -1)
        x_norm = x / x.norm()

        proj_loss = torch.mm(y, x_norm.transpose(0, 1)) / x.norm()
        return proj_loss


# square sum of dot product
class Mid_layer_target_Loss(torch.nn.Module):
    def __init__(self):
        super(Mid_layer_target_Loss, self).__init__()

    def forward(self, old_attack_mid, new_mid, original_mid, coeff):
        x = (old_attack_mid - original_mid).view(1, -1)
        y = (new_mid - original_mid).view(1, -1)

        x_norm = x / x.norm()
        if (y == 0).all():
            y_norm = y
        else:
            y_norm = y / y.norm()
        angle_loss = torch.mm(x_norm, y_norm.transpose(0, 1))
        magnitude_gain = y.norm() / x.norm()
        return angle_loss + magnitude_gain * coeff


"""Return: perturbed x"""
mid_output = None


def ILA(
    model,
    X,
    X_attack,
    y,
    feature_layer,
    niters=10,
    epsilon=0.01,
    coeff=1.0,
    learning_rate=1,
    dataset="cifar10",
    use_Inc_model = False,
    with_projection=True,
):
    """Perform ILA attack with respect to model on images X with labels y

    Args:
        with_projection: boolean, specifies whether projection should happen
        in the attack
        model: torch model with respect to which attacks will be computed
        X: batch of torch images
        X_attack: starting adversarial examples of ILA that will be modified
        to become more transferable
        y: labels corresponding to the batch of images
        feature_layer: layer of model to project on in ILA attack
        niters: number of iterations of the attack to perform
        epsilon: Linf norm of resulting perturbation; scale of images is -1..1
        coeff: coefficient of magnitude loss in ILA attack
        visualize: whether you want to visualize the perturbations or not
        learning_rate: learning rate of the attack
        dataset: dataset the images are from, 'cifar10' | 'imagenet'

    Returns:
        The batch of modified adversarial examples, examples have been
        augmented from X_attack to become more transferable
    """
    X = X.detach()
    X_pert = torch.zeros(X.size()).cuda()
    X_pert.copy_(X).detach()
    X_pert.requires_grad = True

    def get_mid_output(m, i, o):
        global mid_output
        mid_output = o

    h = feature_layer.register_forward_hook(get_mid_output)

    out = model(X)
    mid_original = torch.zeros(mid_output.size()).cuda()
    mid_original.copy_(mid_output)

    out = model(X_attack)
    mid_attack_original = torch.zeros(mid_output.size()).cuda()
    mid_attack_original.copy_(mid_output)

    adversaries = []

    for iter_n in range(niters):
        output_perturbed = model(X_pert)

        # generate adversarial example by max middle layer pertubation
        # in the direction of increasing loss
        if with_projection:
            loss = Proj_Loss()(
                mid_attack_original.detach(), mid_output, mid_original.detach(), coeff
            )
        else:
            loss = Mid_layer_target_Loss()(
                mid_attack_original.detach(), mid_output, mid_original.detach(), coeff
            )

        loss.backward()
        pert = learning_rate * X_pert.grad.detach().sign()

        X_pert = update_adv(X, X_pert, pert, epsilon)
        X_pert.requires_grad = True        

        if (iter_n+1) % 10 == 0:
            adversaries.append(X_pert.clone().detach())

    h.remove()
    return X_pert.clone().detach()
    # return adversaries


def FIA(
    model,
    X,    
    y,    
    feature_layer,
    N=30,
    drop_rate=0.3,
    niters=10,    
    epsilon=0.01,
    learning_rate=1,
    decay=1,
    dataset="cifar10",
    use_Inc_model = False,    
):
    """
        Feature Importance-aware Attack
    """
    X = X.detach()
    X_pert = torch.zeros(X.size()).cuda()
    X_pert.copy_(X).detach()
    X_pert.requires_grad = True

    def get_mid_output(m, i, o):
        global mid_output
        mid_output = o        
    
    def get_mid_grad(m, i, o):
        global mid_grad
        mid_grad = o        

    h = feature_layer.register_forward_hook(get_mid_output)
    h2 = feature_layer.register_full_backward_hook(get_mid_grad)

    '''
        Set Seeds
    '''
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    '''
        Gradient Aggregate
    '''
    agg_grad = 0
    for _ in range(N):        
        X_random = torch.zeros(X.size()).cuda()
        X_random.copy_(X).detach()
        X_random.requires_grad = True
        Mask = torch.bernoulli(torch.ones_like(X_random)*(1-drop_rate))
        X_random = X_random * Mask
        output_random = model(X_random)        
        loss = 0
        for batch_i in range(X.shape[0]):
            loss += output_random[batch_i][y[batch_i]]        
        model.zero_grad()
        loss.backward()        
        agg_grad += mid_grad[0].detach()    
    for batch_i in range(X.shape[0]):
        agg_grad[batch_i] /= agg_grad[batch_i].norm(2)
    h2.remove()   

    adversaries = []

    momentum = 0
    for iter_n in range(niters):
        output_perturbed = model(X_pert)
        loss = (mid_output * agg_grad).sum()
        model.zero_grad()
        loss.backward()

        # momentum = decay * momentum + X_pert.grad / torch.sum(torch.abs(X_pert.grad))
        # pert = -learning_rate * momentum.sign()

        # No Momentum
        pert = -learning_rate * X_pert.grad.detach().sign()

        X_pert = update_adv(X, X_pert, pert, epsilon)
        X_pert.requires_grad = True        

        if (iter_n+1) % 10 == 0:
            adversaries.append(X_pert.clone().detach())

    h.remove()        
    # return adversaries    
    return X_pert.clone().detach()


def AA(
    model,
    X,    
    y,  
    X_target,  
    feature_layer,        
    niters=10,    
    epsilon=0.01,
    learning_rate=1,
    decay=1,
    dataset="cifar10",
    use_Inc_model = False,    
):
    """
        Activation Attack
    """
    batch_size = X.shape[0]
    X = X.detach()
    X_target = X_target.detach()
    X_pert = torch.zeros(X.size()).cuda()
    X_pert.copy_(X).detach()
    X_pert.requires_grad = True

    def get_mid_output(m, i, o):
        global mid_output
        mid_output = o            

    h = feature_layer.register_forward_hook(get_mid_output)    

    '''
        Set Seeds
    '''
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    '''
     Select target for each input
    '''
    with torch.no_grad():
        model(X)
        X_mid_output = mid_output.clone().detach()
        model(X_target)
        X_target_mid_output = mid_output.clone().detach()
    target_slice = []
    for X_i in X_mid_output:
        max_JAA = 0
        max_slice = 0
        for slice_i, X_target_i in enumerate(X_target_mid_output):
            JAA = (X_i-X_target_i).norm(2).item()
            if JAA>max_JAA:
                max_JAA = JAA
                max_slice = slice_i
        target_slice.append(X_target_mid_output[max_slice].clone().detach())
    targets = torch.stack(target_slice)


    adversaries = []

    momentum = 0
    for iter_n in range(niters):
        output_perturbed = model(X_pert)
        loss = (mid_output - targets).reshape(batch_size, -1).norm(2, dim=1).sum()
        model.zero_grad()
        loss.backward()

        # momentum = decay * momentum + X_pert.grad / torch.sum(torch.abs(X_pert.grad))
        # pert = learning_rate * momentum.sign()

        # No Momentum
        pert = -learning_rate * X_pert.grad.detach().sign()

        X_pert = update_adv(X, X_pert, pert, epsilon)
        X_pert.requires_grad = True        

        if (iter_n+1) % 10 == 0:
            adversaries.append(X_pert.clone().detach())

    h.remove()        
    # return adversaries    
    return X_pert.clone().detach()


def NAA(
    model,
    X,    
    y,    
    feature_layer,
    N=30,    
    niters=10,    
    epsilon=0.01,
    learning_rate=1,
    decay=1,        
    dataset="cifar10",
    use_Inc_model = False,    
):
    """
        NAA attack with default Linear Transformation Functions and Weighted Attribution gamma = 1
    """
    X = X.detach()
    X_pert = torch.zeros(X.size()).cuda()
    X_pert.copy_(X).detach()
    X_pert.requires_grad = True

    def get_mid_output(m, i, o):
        global mid_output
        mid_output = o        
    
    def get_mid_grad(m, i, o):
        global mid_grad
        mid_grad = o        

    h = feature_layer.register_forward_hook(get_mid_output)
    h2 = feature_layer.register_full_backward_hook(get_mid_grad)

    '''
        Set Seeds
    '''
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    '''
        Gradient Aggregate
    '''
    agg_grad = 0
    for iter_n in range(N): 
        X_Step = torch.zeros(X.size()).cuda()
        X_Step = X_Step + X.clone().detach() * iter_n / N
        
        output_random = model(X_Step)        
        output_random  = torch.softmax(output_random, 1)

        loss = 0
        for batch_i in range(X.shape[0]):
            loss += output_random[batch_i][y[batch_i]]        
        model.zero_grad()
        loss.backward()        
        agg_grad += mid_grad[0].detach()
    agg_grad /= N    
    h2.remove()   

    adversaries = []

    X_prime = torch.zeros(X.size()).cuda()
    model(X_prime)
    Output_prime = mid_output.detach().clone()

    momentum = 0
    for iter_n in range(niters):
        output_perturbed = model(X_pert)
        loss = ((mid_output - Output_prime) * agg_grad).sum()
        model.zero_grad()
        loss.backward()

        # momentum = decay * momentum + X_pert.grad / torch.sum(torch.abs(X_pert.grad))
        # pert = -learning_rate * momentum.sign()

        # No Momentum
        pert = -learning_rate * X_pert.grad.detach().sign()

        X_pert = update_adv(X, X_pert, pert, epsilon)
        X_pert.requires_grad = True        

        if (iter_n+1) % 10 == 0:
            adversaries.append(X_pert.clone().detach())

    h.remove()        
    # return adversaries    
    return X_pert.clone().detach()


def load_model(model_name):    
    if model_name == 'ResNet50':
        return torchvision.models.resnet50(pretrained=True).cuda()
    elif model_name == 'DenseNet121':
        return torchvision.models.densenet121(pretrained=True).cuda()    
    elif model_name == 'VGG19':
        return torchvision.models.vgg19_bn(pretrained=True).cuda()
    elif model_name == 'Inc-v3':
        return torchvision.models.inception_v3(pretrained=True).cuda()
    else:
        print('Not supported model')

def get_source_layers(model_name, model):
    if model_name == 'ResNet18':
        # exclude relu, maxpool
        return list(enumerate(map(lambda name: (name, model._modules.get(name)), ['conv1', 'bn1', 'layer1', 'layer2','layer3','layer4','fc'])))
    
    elif model_name == 'ResNet50':
        # exclude relu, maxpool
        return list(enumerate(map(lambda name: (name, model._modules.get(name)), ['conv1', 'layer1', 'layer2','layer3','layer4', 'fc'])))
    
    elif model_name == 'DenseNet121':
        # exclude relu, maxpool
        layer_list = list(map(lambda name: (name, model._modules.get('features')._modules.get(name)), ['conv0', 'denseblock1', 'transition1', 'denseblock2', 'transition2', 'denseblock3', 'transition3', 'denseblock4', 'norm5']))
        layer_list.append(('classifier', model._modules.get('classifier')))
        return list(enumerate(layer_list))
                                             
    elif model_name == 'SqueezeNet1.0':
        # exclude relu, maxpool
        layer_list = list(map(lambda name: ('layer '+name, model._modules.get('features')._modules.get(name)), ['0','3','4','5','7','8','9','10','12']))
        layer_list.append(('classifier', model._modules.get('classifier')._modules.get('1')))
        return list(enumerate(layer_list))
    
    elif model_name == 'alexnet':
        # exclude avgpool
        layer_list = list(map(lambda name: ('layer '+name, model._modules.get('features')._modules.get(name)), ['0','3','6','8','10']))
        layer_list += list(map(lambda name: ('layer '+name, model._modules.get('classifier')._modules.get(name)), ['1','4','6']))
        return list(enumerate(layer_list))
    
    elif model_name == 'IncRes-v2':
        # exclude relu, maxpool
        return list(enumerate(map(lambda name: (name, model._modules.get(name)), ['conv2d_1a', 'conv2d_2a', 'conv2d_2b', 'maxpool_3a', 'conv2d_3b', 'conv2d_4a', 'maxpool_5a', 'mixed_5b', 'repeat', 'mixed_6a','repeat_1', 'mixed_7a', 'repeat_2', 'block8', 'conv2d_7b', 'avgpool_1a', 'last_linear'])))

    elif model_name == 'Inc-v4':
        # exclude relu, maxpool
        layer_list = list(map(lambda name: (name, model._modules.get('features')._modules.get(name)), ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']))
        return list(enumerate(layer_list))
                                             
    elif model_name == 'Inc-v3':
        # exclude relu, maxpool
        layer_list = list(map(lambda name: (name, model._modules.get(name)), ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c']))
        return list(enumerate(layer_list))
    
    else:
        # model is not supported
        assert False

def run_attack(attack, use_Inc_model = False):
    # load source model    
    model = load_model(opt.modeltype)
    model.eval()
    model_name = opt.modeltype

    # load transfer models
    all_model_names = ['ResNet50', 'DenseNet121', 'Inc-v3', 'VGG19']
    transfer_model_names = [x for x in all_model_names if x != opt.modeltype]
    
    transfer_models = [load_model(x) for x in transfer_model_names]
    for model_ in transfer_models:
        model_.eval()
    
    success_rate = dict()
    for name in all_model_names:
        success_rate[name] = 0

    print('Loaded model...')
    # pre-process input image
    mean, stddev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform_resize_224 = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    transform_resize_299 = transforms.Compose([transforms.Resize(299), transforms.CenterCrop(299)])
    transform_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, stddev)])
    invTrans = NormalizeInverse(mean, stddev)
    toPIL = transforms.ToPILImage()
    source_layers = get_source_layers(opt.modeltype, model)

    # image data and labels are packed with Pickle as a dictionary
    # key: true label, value: the list of images corresponding to the true label
    with open('data.pickle', 'rb') as f:
        image_dict = pickle.load(f)
        
    print('Image Loaded')
    adv_images_dict = dict()    
    for true_label in tqdm(image_dict.keys()):        
        # print(true_label)        
        image_batch = image_dict[true_label]        
        label_batch = [true_label]*len(image_batch)
        if use_Inc_model:
            image_batch = [transform_resize_299(image_) for image_ in image_batch]
        else:
            image_batch = [transform_resize_224(image_) for image_ in image_batch]

        image_batch = torch.stack([transform_norm(image_) for image_ in image_batch]).cuda()
        label_batch = torch.tensor(label_batch).cuda()                   

        if attack is ILA:
            ifgsm_guide = ifgsm(model, image_batch, label_batch, learning_rate=2./255., epsilon=opt.epsilon, niters=10, dataset='imagenet')
            # fgsm_guide = fgsm(model, image_batch, label_batch, epsilon=opt.epsilon, dataset='imagenet')
            adversaries = ILA(model, image_batch, ifgsm_guide, label_batch, source_layers[opt.layerindex][1][1], learning_rate=opt.learning_rate, epsilon=opt.epsilon, niters=opt.niters, dataset='imagenet', use_Inc_model=use_Inc_model)
        elif attack is FIA:
            adversaries = FIA(model, image_batch, label_batch, source_layers[opt.layerindex][1][1], learning_rate=opt.learning_rate, epsilon=opt.epsilon, niters=opt.niters, dataset='imagenet', use_Inc_model=use_Inc_model)
        elif attack is AA:
            # ImageNet 1000 classes
            target_classes = list(range(1000))
            target_classes.remove(true_label)            
            target_class = random.sample(target_classes, k=4)            
            target_batch = []
            for class_i in target_class:
                target_batch += image_dict[class_i]
            if use_Inc_model:
                target_batch = [transform_resize_299(image_) for image_ in target_batch]
            else:
                target_batch = [transform_resize_224(image_) for image_ in target_batch]
            target_batch = torch.stack([transform_norm(image_) for image_ in target_batch]).cuda()
            adversaries = AA(model, image_batch, label_batch, target_batch, source_layers[opt.layerindex][1][1], learning_rate=opt.learning_rate, epsilon=opt.epsilon, niters=opt.niters, dataset='imagenet', use_Inc_model=use_Inc_model)
        elif attack is Transferable_Adversarial_Perturbations:
            adversaries = Transferable_Adversarial_Perturbations(model, image_batch, label_batch, learning_rate=opt.learning_rate, epsilon=opt.epsilon, niters=opt.niters, dataset='imagenet', use_Inc_model=use_Inc_model)        
        elif attack is NAA:
            adversaries = NAA(model, image_batch, label_batch, source_layers[opt.layerindex][1][1], learning_rate=opt.learning_rate, epsilon=opt.epsilon, niters=opt.niters, dataset='imagenet', use_Inc_model=use_Inc_model)            
        
        adv_images_dict[true_label] = []
        for adv_img in adversaries:
            adv_images_dict[true_label].append(toPIL(invTrans(adv_img)))
        
        output = model(adversaries).max(dim=1)[1]
        success_rate[model_name] += (output!=label_batch).sum().item()

        for transfer_model_name, transfer_model in zip(transfer_model_names, transfer_models):
            output = transfer_model(adversaries).max(dim=1)[1]
            success_rate[transfer_model_name]+=(output!=label_batch).sum().item()
        # break

    for model_name_ in success_rate.keys():
        print('Model: %s Success Rate:%f' % (model_name_, success_rate[model_name_] / 5000.))
    # with open(opt.filename+'.txt', 'w') as f:
    #     f.write(str(success_rate))
    # print(success_rate)
    with open(opt.filename+'.pickle', 'wb') as f:
        pickle.dump(adv_images_dict, f)
    return adv_images_dict

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return str(self.__dict__)

## Demos
if __name__ == '__main__':
    opt = Namespace(modeltype='ResNet50', layerindex=2, filename='ResNet50_ILA_16_50_2_2_ifgsm', niters=50, epsilon=16./255., learning_rate=2./255.)
    adv_image_dict = run_attack(ILA)
    # opt = Namespace(modeltype='ResNet50', layerindex=2, filename='ResNet50_FIA_16_50_2_2', niters=50, epsilon=16./255., learning_rate=2./255.)
    # adv_image_dict = run_attack(FIA)
    # opt = Namespace(modeltype='ResNet50', layerindex=2, filename='ResNet50_TAP_16_50_2_2_Original', niters=50, epsilon=16./255., learning_rate=2./255.)
    # adv_image_dict = run_attack(Transferable_Adversarial_Perturbations)
    # opt = Namespace(modeltype='ResNet50', layerindex=2, filename='ResNet50_NAA_16_50_2_2', niters=50, epsilon=16./255., learning_rate=2./255.)
    # adv_image_dict = run_attack(NAA)    
    # opt = Namespace(modeltype='ResNet50', layerindex=2, filename='ResNet50_AA_16_50_2_2_Momentum', niters=50, epsilon=16./255., learning_rate=2./255.)
    # adv_image_dict = run_attack(AA)

    # for key in adv_image_dict:
    #     for image_ in adv_image_dict[key]:
    #         image_.show()
    pass