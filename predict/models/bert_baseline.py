# coding: utf-8

import sys
import torch
import random
import logging
import numpy as np
from torch import nn
from torch.autograd import Variable
from predict.utils import runBiRNN, sequence_mask, fix_hidden

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile='full')


class Baseline(nn.Module):
    def __init__(self, args):
        super(Baseline, self).__init__()
        # args
        self.args = args
        self.hidden_size = args.hidden_size
        # embedding
        self.token_embedding = nn.Embedding(args.vocab_size, args.word_dim)
        if args.embed_matrix is not None:
            self.token_embedding.weight = nn.Parameter(torch.FloatTensor(args.embed_matrix))
        # token_lstm
        self.token_lstm = nn.LSTM(args.word_dim, args.hidden_size, bidirectional=True, batch_first=True,
                               num_layers=args.num_layers, dropout=args.dropout_p)
        # softmax
        self.out = nn.Linear(2 * self.args.hidden_size, self.args.class_num)

    def forward(self, inputs):
        # unpack inputs to data; _, (batch_size)
        tokenize, tokenize_len = inputs
        tokenize, tokenize_len = tokenize.to(self.args.device), tokenize_len.to(self.args.device)
        tokenize_embed = self.token_embedding(tokenize)
        token_out, token_hidden = runBiRNN(self.token_lstm, tokenize_embed, tokenize_len, total_length=self.args.max_len)
        token_hidden = fix_hidden(token_hidden[0]).squeeze(0)
        score = self.out(token_hidden)
        return score
