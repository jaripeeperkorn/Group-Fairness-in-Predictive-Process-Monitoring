import torch.nn as nn
import os
import torch
import wandb
import logging
import numpy as np
logging.getLogger().setLevel(logging.INFO)
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

import torch.nn.utils.rnn as rnn_utils

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

#! we need to know how many categorical features to know how many embedding layers?
class LSTM_Model(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout, lstm_size, max_length):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.lstm_dropout = dropout
        self.lstm_size= lstm_size
        self.embed_size = embed_size
        self.embed_act = nn.Embedding(self.vocab_size[0], self.embed_size)
        self.embed_res = nn.Embedding(self.vocab_size[1], self.embed_size)
        self.lstm = nn.LSTM(2*self.embed_size, self.embed_size, dropout = self.lstm_dropout, num_layers=self.lstm_size, batch_first=True)
        self.final_output = nn.Linear(self.embed_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x_act, x_res):
        batch_size = x_act.size(0)  # Get the batch size
        #hidden_state = torch.zeros(self.lstm_size, batch_size, self.embed_size).to(self.device)
        #cell_state = torch.zeros(self.lstm_size, batch_size, self.embed_size).to(self.device)

        x_act_embed_enc = self.embed_act(x_act).to(self.device)
        x_res_embed_enc = self.embed_res(x_res).to(self.device)
        x_embed_enc = torch.cat([x_act_embed_enc, x_res_embed_enc], dim=2)
        l1_out, _ = self.lstm(x_embed_enc)
        output = l1_out[:, -1, :]
        output = self.final_output(output)
        output = torch.sigmoid(output)
        return output


