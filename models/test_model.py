import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, hidden_size=64):
        super(BaseModel, self).__init__()
        self.hidden_size = hidden_size

        # img
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 64 * 64, hidden_size)

    def forward(self, img_tens):
        # text_feat = self.fc2(self.flatten(text_tens))

        img_feat = self.conv1(img_tens)
        img_feat = self.maxpool(img_feat)
        img_feat = self.conv2(img_feat)
        img_feat = self.fc1(self.flatten(img_feat))
        return img_feat