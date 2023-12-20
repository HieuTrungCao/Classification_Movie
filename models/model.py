import torch.nn as nn
import torch
import torchvision.models as models

from .lstm import LSTM
from .resnet import resnet50
from .img_model import ImgModel

class Model(nn.Module):
    def __init__(self, num_class, pretrained, hidden_state_img = 512, use_title=False, hidden_state_title = None):
        super(Model, self).__init__()

        self.use_title = use_title
        # if use_title:
        #     self.img_model = ImgModel(hidden_state_img)
        #     self.input_dim = hidden_state_img
        #     self.text_model = LSTM()
        #     self.input_dim += hidden_state_title
        #     self.linear = nn.Linear(self.input_dim, num_class)
        # else:
            # self.img_model = ImgModel(num_class)
        self.img_model = models.vgg16(pretrained)
        self.linear = nn.Linear(1000, num_class)
        
        
    def forward(self, img, title):
        
        if not self.use_title:
            out = self.img_model(img)
            out = self.linear(out)
            return out        

        out = self.img_model(img)
        t = self.text_model(title)
        out = torch.concatenate((out, t), dim = 1)

        return out