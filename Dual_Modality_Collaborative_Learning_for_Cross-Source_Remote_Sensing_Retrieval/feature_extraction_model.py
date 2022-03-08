import torch
import math
from torch import nn 
from torch.nn.parameter import Parameter
import torch.nn.functional as F
device = torch.device('cuda:1')

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride = 1,
                downsample = None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,stride = stride,
                               padding = 1,bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,kernel_size=1,bias = False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual  = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block = Bottleneck,layers = [3,4,6,3]):
        super(ResNet, self).__init__()
        self.convert_channel = nn.Sequential(
            nn.Conv2d(4, 3,kernel_size = 1,stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace = True)
        )
        self.dimension = nn.Sequential(
            nn.Conv2d(2048, 512,kernel_size = 1,stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True)
        )
        self.dimension1 = nn.Sequential(
            nn.Conv2d(1024, 400,kernel_size = 1,stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(400),
            nn.ReLU(inplace = True)
        )
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                     bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bnLayer = nn.BatchNorm1d(512)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Sequential(
#             nn.Linear(2048*7*7,2048),
#             nn.Linear(200,1000),
#             nn.ReLU(True),
#             nn.Dropout(),
            nn.Linear(400,8)
#             nn.Softmax(dim = 1)
        )
#         self.classi = nn.Linear(512*self.expension,8)
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
            nn.Conv2d(self.in_channels, channels * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)
    def l2_norm(self,inputs, axit=1):
        norm = torch.norm(inputs,2,axit,True)
        output = torch.div(inputs, norm)
        return output 

    def forward(self, x):
        x = self.convert_channel(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_mid = self.layer3(x)
        gap = torch.nn.functional.adaptive_avg_pool2d(x_mid,(1,1))
        gap = self.dimension1(gap)
        gap = gap.view(gap.size(0),-1)

        y = self.l2_norm(gap)
        y_fc = self.fc(y)

        x = self.layer4(x_mid)
#         x = self.avgpool(x)
#         x = self.dimension0(x)
        
#         gap = gap.view(gap.size(0),-1)
#         x = x.view(x.size(0),-1)
#         y = torch.cat((gap,x),1)
#         y = self.l2_norm(y)
#         y_fc = self.fc(y)

#         specific_fea = specific_fea.view(specific_fea.size(0), -1)
#         specific_fea = self.bnLayer(specific_fea)
#         x__ = self.fc(specific_fea)

        return x_mid,y,y_fc
class shared(nn.Module):
    def __init__(self, block = Bottleneck,layers = [3]):
        super(shared, self).__init__()
        self.in_channels = 1024
        self.layer4 = self._make_layer(block, 512, layers[0], stride=2)  
        
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.bnLayer = nn.BatchNorm2d(512)
        self.dimension = nn.Sequential(
            nn.Conv2d(2048, 400,kernel_size = 1,stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(400),
            nn.ReLU(inplace = True)
        )
        
        self.fc = nn.Sequential(
#             nn.Linear(2048*7*7,2048),
#             nn.Linear(200,1000),
#             nn.ReLU(True),
#             nn.Dropout(),
            nn.Linear(400,8)   
        )
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
            nn.Conv2d(self.in_channels, channels * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)
    def l2_norm(self,inputs, axit=1):
        norm = torch.norm(inputs,2,axit,True)
        output = torch.div(inputs, norm)
        return output 
    def forward(self, x, y):
        x = self.layer4(x)
        x = self.dimension(x)
        x = self.avgpool(x)
#         x = self.bnLayer(x)
        x = self.l2_norm(x)
        
        x = x.view(x.size(0), -1)
        x_ = self.fc(x)
        
        y = self.layer4(y)
        y = self.dimension(y)
        y = self.avgpool(y)
#         y = self.bnLayer(y)
        y = self.l2_norm(y)
        y = y.view(y.size(0),-1)
        y_ = self.fc(y)
        return x,y,x_,y_
class W(nn.Module):

    def __init__(self):
        super(W, self).__init__()
        self.conv1 = nn.Conv1d(800, 400, 1)
    def forward(self, x):
        x =  x.unsqueeze(2)
        x = self.conv1(x)
        return x.squeeze()
class T(nn.Module):
    def __init__(self):
        super(T, self).__init__()
        self.fc = nn.Linear(400,8)   
    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x