# coding: utf-8

import sys
import torch
import random
import logging
import numpy as np
from torch import nn
from torch.autograd import Variable
from utils import runBiRNN, sequence_mask

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
        # pos_tag embedding
        self.pos_tag_embedding = nn.Embedding(len(args.pos_tag_vocab), args.word_dim)
        # token_lstm
        self.token_lstm = nn.LSTM(args.word_dim, args.hidden_size, bidirectional=True, batch_first=False,
                               num_layers=args.num_layers, dropout=args.dropout_p)
        # table_encoder for cols and cells
        self.table_encoder = TableRNNEncoder(self.args)
        # point_net_decoder
        self.pointer_net_decoder = PointerNetRNNDecoder(self.args, input_dim=self.args.word_dim)

    # def init_parameters(self):
    #     torch.nn.init.xavier_uniform_(self.decoder_input)
    #     torch.nn.init.xavier_uniform_(self.unk_tensor)

    def forward(self, inputs):
        # unpack inputs to data
        tokenize, tokenize_len = inputs[0]  # _, (batch_size)
        pos_tag = inputs[1][0]
        columns_split, columns_split_len = inputs[2]
        columns_split_marker, columns_split_marker_len = inputs[3]  # _, (batch_size)
        cells_split, cells_split_len = inputs[4]
        cells_split_marker, cells_split_marker_len = inputs[5]  # _, (batch_size)
        batch_size = tokenize.size(0)
        # encode token
        token_embed = self.token_embedding(tokenize)
        token_embed = token_embed.transpose(0, 1).contiguous()  # (tokenize_max_len, batch_size, word_dim)
        # add pos_tag on token_embed
        pos_tag_embed = self.pos_tag_embedding(pos_tag).transpose(0, 1).contiguous()  # (tokenize_max_len, batch_size, word_dim)
        token_embed += pos_tag_embed
        # run token lstm
        token_out, token_hidden = runBiRNN(self.token_lstm, token_embed, tokenize_len, total_length=self.args.tokenize_max_len)  # (tokenize_max_len, batch_size, 2*hidden_size), _
        # encode columns
        col_embed = self.token_embedding(columns_split).transpose(0, 1).contiguous()  # (columns_token_max_len, batch_size, word_dim)
        col_out, col_hidden = self.table_encoder(self.token_lstm, col_embed, columns_split_len, columns_split_marker, hidden=token_hidden, total_length=self.args.columns_token_max_len)  # (columns_split_marker_max_len - 1, batch_size, 2 * hidden_size)
        # encode cells
        cell_embed = self.token_embedding(cells_split).transpose(0,1).contiguous()
        cell_out, cell_hidden = self.table_encoder(self.token_lstm, cell_embed, cells_split_len, cells_split_marker, hidden=col_hidden, total_length=self.args.cells_token_max_len)
        # concat as memory_bank
        memory_bank = torch.cat([token_out, col_out, cell_out], dim=0).transpose(0, 1).contiguous()
        # decode one step (encode)
        pointer_align_scores, _, _ = self.pointer_net_decoder(tgt=token_embed, src=memory_bank, hidden=col_hidden,
                                                                tgt_lengths=tokenize_len,
                                                                tgt_max_len=self.args.tokenize_max_len,
                                                                src_lengths=None,
                                                                src_max_len=None)
        logger.debug('pointer_align_scores'), logger.debug(pointer_align_scores.size())
        # (batch_size, tgt_len, src_len)
        return pointer_align_scores
