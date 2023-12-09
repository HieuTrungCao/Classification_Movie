import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, hidden_state) -> None:
        super(LSTM, self).__init__()
        self.hidden_state = hidden_state

    def forward(self, input):
        return input