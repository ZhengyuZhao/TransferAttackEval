import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
import json
import pickle

import pyutils.io as io

class Preprocessing_Layer(torch.nn.Module):
    def __init__(self, mean, std):
        super(Preprocessing_Layer, self).__init__()
        self.mean = mean
        self.std = std

    def preprocess(self, img, mean, std):
        img = img.clone()
        #img /= 255.0

        img[:,0,:,:] = (img[:,0,:,:] - mean[0]) / std[0]
        img[:,1,:,:] = (img[:,1,:,:] - mean[1]) / std[1]
        img[:,2,:,:] = (img[:,2,:,:] - mean[2]) / std[2]

        #img = img.transpose(1, 3).transpose(2, 3)
        return(img)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        res = self.preprocess(x, self.mean, self.std)
        return res

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ImagePathDataset(VisionDataset):
    def __init__(self, config, transform=None, target_transform=None,
                 loader=default_loader, return_paths=False):
        super().__init__(root=config["root"], transform=transform, target_transform=target_transform)
        self.config = config

        self.loader = loader
        self.extensions = IMG_EXTENSIONS

        self.classes = config["classes"]
        self.class_to_idx = config["class_to_idx"]
        self.samples = config["samples"]
        self.targets = [s[1] for s in self.samples]
        self.return_paths = return_paths

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        output = sample, target

        if self.return_paths:
            return output, path
        else:
            return output

    def __len__(self):
        return len(self.samples)

    @classmethod
    def from_path(cls, config_path, *args, **kwargs):
        return cls(config=io.read_json(config_path), *args, **kwargs)


def set_requires_grad(named_parameters, requires_grad):
    for name, param in named_parameters:
        param.requires_grad = requires_grad



def save_images(images, img_list, idx, output_dir):
    """Saves images to the output directory.
        Args:
          images: tensor with minibatch of images
          img_list: list of filenames 
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
          output_dir: directory where to save images
    """
    for i in range(idx):
        filename = os.path.basename(img_list[i])
        cur_images = (images[i, :, :, :].transpose(1, 2, 0) * 255).astype(np.uint8)

        im = Image.fromarray(cur_images)
        im.save('{}.png'.format(os.path.join(output_dir, filename[:-5])))

def generate_data_pickle():
    all_image = dict()
    label_dict = dict()

    with open('../imagenet_class_index.json') as f:
        data = json.load(f)    
        for i in data.keys():
            label_dict[data[i][0]] = int(i)        

    for root, dirs, files in os.walk('../val5000'):
        for name in dirs:
            dir_name = os.path.join(root, name)        
            image_list = []
            for root_, dirs_, files_ in os.walk(dir_name):                
                for name_ in files_:
                    obj = Image.open(os.path.join(root_, name_))
                    obj = obj.convert('RGB')                                                            
                    image_list.append(obj)                                    
            all_image[label_dict[name]] = image_list                                    

    with open('data.pickle', 'wb') as f:
        pickle.dump(all_image, f, pickle.HIGHEST_PROTOCOL)