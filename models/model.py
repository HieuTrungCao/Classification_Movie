import torch.nn as nn
import torch
import torchvision.models as models

from .lstm import LSTM
from .resnet import resnet50
from .img_model import ImgModel

class Model(nn.Module):
    def __init__(self, num_class, pretrained, title_length, num_layers, hidden_state_img = 512, use_title=False, hidden_state_title = None):
        super(Model, self).__init__()

        self.use_title = use_title
        if use_title:
            self.img_model = models.vgg16(pretrained)
            self.input_dim = hidden_state_title + 1000
            self.text_model = nn.LSTM(title_length, hidden_state_title, num_layers, batch_first=True)
            self.input_dim += hidden_state_title
            self.linear = nn.Linear(self.input_dim, num_class)
        else:
        
            self.img_model = models.vgg16(pretrained)
            for param in self.img_model.parameters():
                param.requires_grad = False

            self.img_model.classifier.requires_grad = True
        
            self.linear = nn.Linear(1000, num_class)
        
        
    def forward(self, img, title):
        
        if not self.use_title:
            out = self.img_model(img)
            out = self.linear(out)
            return out        

        out = self.img_model(img)
        t = self.text_model(title)
        out = torch.cat((out, t), dim = 1)
        out = self.linear(out)
        return out