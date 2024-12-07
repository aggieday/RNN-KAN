import torch.nn as nn
from kan import KAN

# define the RNN-KAN model
class RNN_KANModel(nn.Module):
    def __init__(self, input_size, hidden_size, kan_hidden_size, output_size, grid_size, num_layers=1):
        super(RNN_KANModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.kan = KAN([hidden_size, kan_hidden_size, output_size], grid_size) # layers_hidden, grid_size=5, spline_order=3

    def forward(self, x):
        # the shape of x (batch_size, sequence_length, input_size)
        # the shape of out (batch_size, sequence_length, hidden_size)
        # the shape of hn (num_layers, batch_size, hidden_size)
        out, hn = self.rnn(x)  
        out = self.kan(hn[-1])
        return out