import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModel

class Model(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, pretrained):
    super(Model, self).__init__()
    # self.img_model = models.vgg16(pretrained)

    # for param in self.img_model.features.parameters():
    #   param.requires_grad = False

    self.img_model = models.resnet50()

    for param in self.img_model.parameters():
      param.requires_grad = False

    self.img_model.avgpool.requires_grad = True
    self.img_model.fc.requires_grad = True

    self.embed = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
    self.fc_lstm = nn.Linear(hidden_dim, 128)

    self.fc1 = nn.Linear(1000, 64)
    self.fc2 = nn.Linear(64, num_classes)

    self.dropout = nn.Dropout(0.2)

  def forward(self, image_tensor, title_tensor):
    cnn = self.img_model(image_tensor)

    # lstm = self.embed(title_tensor)
    # lstm, (hidden, cell) = self.lstm(lstm)
    # lstm_out = self.fc_lstm(hidden[-1])

    out = F.relu(self.fc1(cnn))
    out = self.fc2(out)

    return out
  
class ModelWithBert(nn.Module):
  def __init__(self, num_classes, title_model):
    super(ModelWithBert, self).__init__()
    # self.img_model = models.vgg16(pretrained)

    # for param in self.img_model.features.parameters():
    #   param.requires_grad = False

    self.img_model = models.resnet50()

    for param in self.img_model.parameters():
      param.requires_grad = False

    self.img_model.avgpool.requires_grad = True
    self.img_model.fc.requires_grad = True

    self.title_model = AutoModel.from_pretrained(title_model)

    # for param in self.title_model.parameters():
    #   param.requires_grad = False

    self.dropout = nn.Dropout(p=0.1, inplace=False)
  
    self.fc1 = nn.Linear(1768, 64)
    self.fc2 = nn.Linear(64, num_classes)

  def forward(self, image_tensor, title_tensor):
    cnn = self.img_model(image_tensor)
    # , token_type_ids=title_tensor['token_type_ids']
    title = self.title_model(input_ids = title_tensor["input_ids"], attention_mask=title_tensor["attention_mask"])
    title = self.dropout(title.last_hidden_state[:, 0, :])

    out = self.fc1(torch.concat([cnn, title], dim=1))
    out = F.relu(out)
    out = self.fc2(out)
    return out