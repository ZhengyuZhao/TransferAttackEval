import torch
from torch import Tensor
import torch.nn as nn

from typing import Type, Any, Callable, Union, List, Optional

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

###LOAD Pretrained Model#####
# import pretrainedmodels
# net = pretrainedmodels.__dict__["resnet50"](num_classes=1000, pretrained='imagenet')
# model = resnet50()
# model_dict = model.state_dict()
# pre_dict = net.state_dict()
# state_dict = {k:v for k,v in pre_dict.items() if k in model_dict.keys()}
# print("Loaded pretrained weight. Len :",len(pre_dict.keys()),len(state_dict.keys()))  
# model_dict.update(state_dict)
# model.load_state_dict(model_dict)     
# model = nn.Sequential(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), model)        
# model = model.cuda().eval()
##############################



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=nn.LeakyReLU(negative_slope=1),
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        beta_value: float = 1.0,
        decay_value: float = 1.0,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.beta = beta_value
        if self.beta>0 :
            self.nonlin = nn.Softplus(beta=beta_value)
        self.decay = decay_value
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.beta<0 :
            out = self.relu(out)
        else:
            out = self.nonlin(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(x)

        out = out*self.decay + identity
        if self.beta<0 :
            out = self.relu(out)
        else:
            out = self.nonlin(out)

        return out

class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = nn.LeakyReLU(negative_slope=1),
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        beta_value: float = 1.0,
        decay_value: float = 1.0,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.beta = beta_value
        if self.beta>0 :
            self.nonlin = nn.Softplus(beta=beta_value)
        self.decay = decay_value
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.beta<0 :
            out = self.relu(out)
        else:
            out = self.nonlin(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.beta<0 :
            out = self.relu(out)
        else:
            out = self.nonlin(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)
        out = out*self.decay + identity
        if self.beta<0 :
            out = self.relu(out)
        else:
            out = self.nonlin(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        decays: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        beta_value: float = 1.0
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.beta_value = beta_value
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.beta = beta_value
        if self.beta>0 :
            self.nonlin = nn.Softplus(beta=beta_value)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],decay=decays[0])
        self.layer2 = self._make_layer(block, 128, layers[1], decay=decays[1],stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], decay=decays[2],stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], decay=decays[3],stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        #self.last_linear = nn.Linear(512 * block.expansion, num_classes)
        print("Load Resnet for IAA!")
        self.maxpool2= nn.AdaptiveAvgPool2d((2, 2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False,decay:float=1.0) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
                nn.LeakyReLU(negative_slope=1),
            )
        else:
            downsample = nn.Sequential(nn.LeakyReLU(negative_slope=1))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,self.beta_value,decay_value=decay))
        self.inplanes = planes * block.expansion
        downsample = nn.Sequential(nn.LeakyReLU(negative_slope=1))
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,downsample=downsample, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,beta_value=self.beta_value,decay_value=decay))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        feature = []

        x = self.conv1(x)
        x = self.bn1(x)
        if self.beta<0 :
            x = self.relu(x)
        else:
            x = self.nonlin(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feature1 = x
        feature.append(feature1)

        x = self.layer2(x)
        feature2 = x
        feature.append(feature2)

        x = self.layer3(x)
        feature3 = x
        feature.append(feature3)

        x = self.layer4(x)
        feature4 = x
        feature.append(feature4)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        score = self.fc(x)
        #score = self.last_linear(x)

        return score #,feature

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    decays: List[int],
    pretrained: bool,
    progress: bool,
    beta_value:float,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers,decays,beta_value=beta_value, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet50(decays=[1.0, 0.85, 0.65, 0.15],pretrained: bool = False, progress: bool = True,beta_value:float=25.0, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], decays,pretrained, progress,beta_value,
                   **kwargs)

def resnext101_32x8d(decays=[1.0, 0.85, 0.65, 0.15],pretrained: bool = False, progress: bool = True,beta_value:float=1.0, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],decays,
                   pretrained, progress,beta_value, **kwargs)


def wide_resnet50_2(decays=[1.0, 0.85, 0.65, 0.15],pretrained: bool = False, progress: bool = True,beta_value:float=25.0, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],decays,
                   pretrained, progress,beta_value, **kwargs)


def wide_resnet101_2(decays=[1.0,1.0,1.0,1.0],pretrained: bool = False, progress: bool = True,beta_value:float=25.0, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],decays,
                   pretrained, progress,beta_value, **kwargs)

def resnet101(decays=[1.0, 0.85, 0.65, 0.15],pretrained: bool = False, progress: bool = True,beta_value:float=25.0, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], decays,pretrained, progress,beta_value,
                   **kwargs)


def resnet152(decays=[1.0, 0.85, 0.65, 0.15],pretrained: bool = False, progress: bool = True,beta_value:float=25.0, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3],decays, pretrained, progress,beta_value,
                   **kwargs)


def resnext50_32x4d(decays=[1.0, 0.85, 0.65, 0.15],pretrained: bool = False, progress: bool = True,beta_value:float=25.0, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],decays,
                   pretrained, progress,beta_value, **kwargs)

def resnet18(beta_value:float=25.0, decays=[1.0, 0.85, 0.65, 0.15],pretrained: bool = False, progress: bool = True,**kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], decays,pretrained, progress,beta_value,
                   **kwargs)


def resnet34(decays=[1.0, 0.85, 0.65, 0.15],pretrained: bool = False, progress: bool = True,beta_value:float=25.0, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3],decays, pretrained, progress,beta_value,
                   **kwargs)

