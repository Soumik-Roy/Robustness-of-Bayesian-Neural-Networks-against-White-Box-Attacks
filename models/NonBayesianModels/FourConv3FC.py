import torch.nn as nn
from layers.misc import FlattenLayer


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        nn.init.normal_(m.weight, mean=0, std=1)
        nn.init.constant(m.bias, 0)

class FourConvThreeFC(nn.Module):
    def __init__(self, outputs, inputs,activation_type=None):
        super(FourConvThreeFC, self).__init__()
        if activation_type=='softplus' or activation_type==None:
            self.act = nn.Softplus()
        elif activation_type=='relu':
            self.act = nn.ReLU(inplace=False)
        else:
            raise ValueError("Only softplus or relu supported")
        self.features = nn.Sequential(
            nn.Conv2d(inputs, 32, 5, stride=1, padding=2),
            self.act,
            nn.Conv2d(32, 64, 7, stride=1, padding=2),
            self.act,
            nn.Conv2d(64, 128, 7, stride=1, padding=1),
            self.act,
            nn.Conv2d(128, 32, 3, stride=1, padding=2),
            self.act,
            nn.MaxPool2d(4),
        )
        self.classifier = nn.Sequential(
            FlattenLayer(15 * 15 * 32),
            nn.Linear(15 * 15 * 32, 2048),
            self.act,
            nn.Linear(2048, 2048),
            self.act,
            nn.Linear(2048, outputs)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
