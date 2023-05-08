import torch.nn as nn
import torch
import torch.nn.functional as F

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        nn.init.normal_(m.weight, mean=0, std=1)
        nn.init.constant(m.bias, 0)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,activation='softplus'):
        super(BasicBlock, self).__init__()
        if activation=='softplus' or activation==None:
            self.act = nn.Softplus()
        elif activation=='relu':
            self.act = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = self.act
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
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

class ResNet34(nn.Module):
    def __init__(self, block, layers,num_classes,
                   inputs, activation):
        super(ResNet34, self).__init__()
        self.inputs = inputs
        self.inplanes = 64
        if activation=='softplus' or activation==None:
            self.act = nn.Softplus()
        elif activation=='relu':
            self.act = nn.ReLU(inplace=False)
        else:
            raise ValueError("Only softplus or relu supported")
        in_dim = inputs
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = self.act
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],activation=activation)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,activation=activation)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,activation=activation)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,activation=activation)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1,activation='softplus'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,activation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,activation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #if self.inputs == 3:
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        #probas = F.softmax(logits, dim=1)
        return logits
    
def resnet34(outputs, inputs, activation):
    model = ResNet34(block=BasicBlock, 
                   layers=[3, 4, 6, 3],
                   num_classes=outputs,
                   inputs=inputs, activation=activation)
    return model
