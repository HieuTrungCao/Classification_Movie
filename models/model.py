import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models

class Model(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, pretrained):
    super(Model, self).__init__()
    self.img_model = models.vgg16(pretrained)

    for param in self.img_model.features.parameters():
      param.requires_grad = False

    self.embed = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
    self.fc_lstm = nn.Linear(hidden_dim, 128)

    self.fc1 = nn.Linear(1128, 64)
    self.fc2 = nn.Linear(64, num_classes)

    self.dropout = nn.Dropout(0.2)

  def forward(self, image_tensor, title_tensor):
    cnn = self.img_model(image_tensor)

    lstm = self.embed(title_tensor)
    lstm, (hidden, cell) = self.lstm(lstm)
    lstm_out = self.fc_lstm(hidden[-1])

    out = F.relu(self.fc1(torch.concat([cnn, lstm_out], dim=1)))
    out = self.fc2(out)

    return out