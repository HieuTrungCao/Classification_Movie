import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.residual = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        r = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        r = self.residual(r)
        x = x + r
        return self.relu(x)

class Retnet(nn.Module):
    def __init__(self, hidden_state):
        super(Retnet, self).__init__()
        self.layers = [3, 4, 6, 3]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(64, 128, self.layers[0])
        self.layer2 = self.make_layer(128, 256, self.layers[1])
        self.layer3 = self.make_layer(256, 512, self.layers[2])
        self.layer4 = self.make_layer(512, 512, self.layers[3])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512 * 58 * 58, hidden_state)


    def make_layer(self, in_channel, out_channel, nums):
        layers = []
        for i in range(nums - 1):
            layers.append(Block(in_channel, in_channel))
        
        layers.append(Block(in_channel, out_channel))

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
        x = self.flatten(x)
        x = self.linear(x)

        return x
