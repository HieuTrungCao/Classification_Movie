# import torch.nn as nn
# import torch
# import torchvision.models as models

# from .lstm import LSTM
# from .resnet import resnet50
# from .img_model import ImgModel

# class Model(nn.Module):
#     def __init__(self, num_class, pretrained, num_layers, vocab_size, hidden_state_img = 512, use_title=False, hidden_state_title = None, embedding_dim=256):
#         super(Model, self).__init__()

#         self.use_title = use_title
#         if use_title:
#             self.img_model = models.vgg16(pretrained)
#             self.input_dim = hidden_state_title + 1000
#             self.embed = nn.Embedding(vocab_size, embedding_dim)
#             self.text_model = nn.LSTM(embedding_dim, hidden_state_title, num_layers, batch_first=True)
#             self.linear = nn.Linear(self.input_dim, num_class)
#         else:
        
#             self.img_model = models.vgg16(pretrained)
#             for param in self.img_model.parameters():
#                 param.requires_grad = False

#             self.img_model.classifier.requires_grad = True
        
#             self.linear = nn.Linear(1000, num_class)
        
        
#     def forward(self, img, title):
        
#         if not self.use_title:
#             out = self.img_model(img)
#             out = self.linear(out)
#             return out        

#         out = self.img_model(img)

#         t_e = self.embed(title)
#         lstm, (t, cell) = self.text_model(t_e)
#         out = torch.concat([out, t[-1]], dim = 1)
#         out = self.linear(out)
#         return out
import torch.nn as nn
import torch
import torch.nn.functional as F

class ModelVip(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes):
    super(ModelVip, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
    self.pool = nn.MaxPool2d(4, 4)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
    self.fc_cnn = nn.Linear(32 * 16 * 16, 128)

    self.embed = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
    self.fc_lstm = nn.Linear(hidden_dim, 128)

    self.fc1 = nn.Linear(2 * 128, 64)
    self.fc2 = nn.Linear(64, num_classes)

    self.dropout = nn.Dropout(0.2)

  def forward(self, image_tensor, title_tensor):
    cnn = self.pool(self.dropout(F.relu(self.conv1(image_tensor))))
    cnn = self.pool(self.dropout(F.relu(self.conv2(cnn))))
    cnn = torch.flatten(cnn, 1)
    cnn = F.relu(self.fc_cnn(cnn))

    lstm = self.embed(title_tensor)
    lstm, (hidden, cell) = self.lstm(lstm)
    lstm_out = self.fc_lstm(hidden[-1])

    out = F.relu(self.fc1(torch.concat([cnn, lstm_out], dim=1)))
    out = self.fc2(out)

    return out