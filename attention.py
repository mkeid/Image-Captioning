import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()

        # Keep parameters
        self.method = method
        self.hidden_size = hidden_size

        # Define layers
        if self.method == 'general':
            self.attention = nn.Linear(self.hidden_size, self.hidden_size)

        elif self.method == 'concat':
            self.attention = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.floatTensor(1, self.hidden_size))

    # Attend all encoded image feature maps conditioned on the previous hidden state of the decoder
    def forward(self, hidden, image_maps):
        n_maps = len(image_maps)

        # Create variables to store the attention energies
        attention_energies = Variable(torch.zeros(n_maps))
        attention_energies = attention_energies.cuda()

        # Calculate energies for each image feature map
        for i in range(n_maps):
            attention_energies[i] = self.score(hidden, image_maps[i].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attention_energies).unsqueeze(0).unsqueeze(0)

    # Calculate the relevance of a particular feature map in respect to the decoder hidden state
    def score(self, hidden, image_maps):

        if self.method == 'dot':
            energy = hidden.dot(image_maps)

        elif self.method == 'general':
            energy = self.attention(image_maps)
            energy = hidden.dot(energy)

        elif self.method == 'concat':
            energy = self.attention(torch.cat((hidden, image_maps), 1))
            energy = self.other.dor(energy)

        return energy
