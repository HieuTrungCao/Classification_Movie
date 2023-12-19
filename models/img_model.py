import torch.nn as nn

class ImgModel(nn.Module):
    def __init__(self, hidden_state_img):
        super(ImgModel, self).__init__()
        self.hidden_state_img = hidden_state_img
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.batch1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3)
        self.avg = nn.AvgPool2d(3, stride=1)
        self.fc = nn.Linear(128 * 54 * 54, self.hidden_state_img)

    def forward(self, img):
        img = self.conv1(img)
        img = self.batch1(img)
        img = self.relu(img)
        img = self.maxpool(img)
        img = self.conv2(img)
        img = self.avg(img)
        img = img.view(img.size(0), -1)
        img = self.fc(img)

        return img

    