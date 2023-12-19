import torch.nn as nn

class ImgModel(nn.Module):
    def __init__(self, hidden_state_img):
        super(ImgModel, self).__init__()
        self.hidden_state_img = hidden_state_img
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.batch1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3)
        self.batch2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=3)
        self.batch3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.avg = nn.AvgPool2d(3, stride=1)
        self.fc = nn.Linear(512 * 16 * 16, self.hidden_state_img)

    def forward(self, img):
        img = self.conv1(img)
        img = self.batch1(img)
        img = self.relu1(img)
        img = self.maxpool(img)
        img = self.conv2(img)
        img = self.batch2(img)
        img = self.relu2(img)
        img = self.conv3(img)
        img = self.batch3(img)
        img = self.relu3(img)
        img = self.conv4(img)
        img = self.avg(img)
        img = img.view(img.size(0), -1)
        img = self.fc(img)

        return img

    