import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

from collections import OrderedDict
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, hidden_state_img, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.cls = nn.Linear(512 * block.expansion, hidden_state_img)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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

        x = self.avgpool(x)
        feature = x.view(x.size(0), -1)
        cls = self.cls(feature)

        return cls

def remove_module_dict(state_dict):
  new_state_dict = OrderedDict()
  for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
  return new_state_dict

def load_weight_from_dict(model, weight_state_dict, param_pair=None, remove_prefix=True):
  if remove_prefix: weight_state_dict = remove_module_dict(weight_state_dict)
  all_parameter = model.state_dict()
  all_weights   = []
  finetuned_layer, random_initial_layer = [], []
  for key, value in all_parameter.items():
    if param_pair is not None and key in param_pair:
      all_weights.append((key, weight_state_dict[ param_pair[key] ]))
    elif key in weight_state_dict:
      all_weights.append((key, weight_state_dict[key]))
      finetuned_layer.append(key)
    else:
      all_weights.append((key, value))
      random_initial_layer.append(key)
  all_weights = OrderedDict(all_weights)
  model.load_state_dict(all_weights)

def resnet50(hidden_state_img, pretrained=True):
  model = ResNet(hidden_state_img)
  if pretrained:
    weights = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
    load_weight_from_dict(model, weights, None, False)
  return model