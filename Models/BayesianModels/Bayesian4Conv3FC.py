import math
import torch.nn as nn
from layers import BBB_Linear, BBB_Conv2d
from layers import BBB_LRT_Linear, BBB_LRT_Conv2d
from layers import FlattenLayer, ModuleWrapper

class BBB4Conv3FC(ModuleWrapper):
    def __init__(self, outputs, inputs, priors, layer_type='lrt', activation_type='softplus'):
        super(BBB4Conv3FC, self).__init__()

        self.num_classes = outputs
        self.layer_type = layer_type
        self.priors = priors

        if layer_type=='lrt':
            BBBLinear = BBB_LRT_Linear
            BBBConv2d = BBB_LRT_Conv2d
        elif layer_type=='bbb':
            BBBLinear = BBB_Linear
            BBBConv2d = BBB_Conv2d
        else:
            raise ValueError("Undefined layer_type")
        
        if activation_type=='softplus':
            self.act = nn.Softplus
        elif activation_type=='relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.conv1 = BBBConv2d(inputs, 32, 5, padding=2, bias=True, priors=self.priors)
        self.act1 = self.act()
        self.conv2 = BBBConv2d(32, 64, 7, padding=2, bias=True, priors=self.priors)
        self.act2 = self.act()
        self.conv3 = BBBConv2d(64, 128, 7, padding=1, bias=True, priors=self.priors)
        self.act3 = self.act()
        self.conv4 = BBBConv2d(128, 32, 3, padding=2, bias=True, priors=self.priors)
        self.act4 = self.act()
        self.pool = nn.MaxPool2d(4)
        self.flatten = FlattenLayer(15 *15 * 32)
        self.fc1 = BBBLinear(15* 15 * 32, 2048, bias=True, priors=self.priors)
        self.act5 = self.act()
        self.fc2 = BBBLinear(2048, 2048, bias=True, priors=self.priors)
        self.act6 = self.act()
        self.fc3 = BBBLinear(2048, outputs, bias=True, priors=self.priors)
