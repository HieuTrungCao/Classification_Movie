import torch.nn as nn
import torch

from .lstm import LSTM
from .retnet import ResNet


class Model(nn.Module):
    def __init__(self, num_class, hidden_state_img = 512, use_title=False, hidden_state_title = None):
        super(Model, self).__init__()

        self.input_dim = hidden_state_img

        self.use_title = use_title

        if use_title:
            self.lstm = LSTM()
            self.input_dim += hidden_state_title
        self.resnet = ResNet(hidden_state_img)
        
        self.linear = nn.Linear(self.input_dim, num_class)

    def forward(self, img, title):
        
        out = self.img_model(img)
        if self.use_title:
            x = self.lstm(title)
            out = torch.concat([out, x], dim=1)
        out = self.linear(out)

        return out