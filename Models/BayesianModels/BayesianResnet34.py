import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from layers import BBB_Linear, BBB_Conv2d
from layers import BBB_LRT_Linear, BBB_LRT_Conv2d
from layers import FlattenLayer, ModuleWrapper


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, priors, stride=1, downsample=None,activation_type='softplus'):
        super(BasicBlock, self).__init__()
        if activation_type=='softplus':
            self.act = nn.Softplus()
        elif activation_type=='relu':
            self.act = nn.ReLU(inplace=True)
        self.priors = priors
        self.conv1 = BBB_Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=True, priors=self.priors)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = self.act
        self.conv2 = BBB_Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=True, priors=self.priors)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet34(ModuleWrapper):
    def __init__(self, priors,block=BasicBlock, 
                   layers=[3, 4, 6, 3],
                   num_classes=10,
                   inputs=3,
                   layer_type='lrt',
                   activation_type='softplus'):
        super(ResNet34, self).__init__()
        self.inplanes = 64
        self.num_classes = num_classes
        self.layer_type = layer_type
        self.priors = priors

        if layer_type=='lrt':
            self.BBBLinear = BBB_LRT_Linear
            self.BBBConv2d = BBB_LRT_Conv2d
        elif layer_type=='bbb':
            self.BBBLinear = BBB_Linear
            self.BBBConv2d = BBB_Conv2d
        else:
            raise ValueError("Undefined layer_type")
        
        if activation_type=='softplus':
            self.act = nn.Softplus()
        elif activation_type=='relu':
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError("Only softplus or relu supported")
    
        in_dim = inputs
        self.conv1 = self.BBBConv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=True,priors=self.priors)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = self.act
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],self.priors,activation_type=activation_type)
        self.layer2 = self._make_layer(block, 128, layers[1],self.priors, stride=2,activation_type=activation_type)
        self.layer3 = self._make_layer(block, 256, layers[2],self.priors, stride=2,activation_type=activation_type)
        self.layer4 = self._make_layer(block, 512, layers[3], self.priors,stride=2,activation_type=activation_type)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.flatten = FlattenLayer(512*block.expansion)
        self.fc = self.BBBLinear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, priors,stride=1,activation_type='softplus'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.BBBConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True,priors=priors),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,priors, stride, downsample,activation_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,priors, 1, None,activation_type))

        return nn.Sequential(*layers)
    

def BBBresnet34(priors,num_classes, inputs, layer_type, activation_type):
    model = ResNet34(priors,block=BasicBlock, 
                   layers=[3, 4, 6, 3],
                   num_classes=num_classes,
                   inputs=inputs,
                   layer_type=layer_type,
                   activation_type=activation_type)
    return model