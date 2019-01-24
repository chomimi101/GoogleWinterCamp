# coding: utf-8

import torch
import logging
import torch.nn as nn
from utils import runBiRNN
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

logger = logging.getLogger('binding')


class TableRNNEncoder(nn.Module):
    def __init__(self, args, split_type='incell', merge_type='cat'):
        super(TableRNNEncoder, self).__init__()
        self.args = args
        self.split_type = split_type
        self.merge_type = merge_type
        self.hidden_size = self.args.hidden_size
        if self.merge_type == 'mlp':
            self.merge = nn.Sequential(
                nn.Linear(4 * self.hidden_size, 2 * self.hidden_size),
                nn.Tanh())

    def forward(self, encoder, tbl, tbl_len, tbl_split, hidden=None, total_length=None):
        """
        Encode table headers.
            :param tbl: header token list
            :param tbl_len: length of token list (num_table_header, batch)
            :param tbl_split: table header boundary list
        """
        tbl_context, hidden = runBiRNN(encoder, tbl, tbl_len, hidden=hidden, total_length=total_length)
        logger.debug('tbl_context')
        logger.debug(tbl_context.size())
        logger.debug(tbl_context)
        # --> (num_table_header, batch, hidden_size * num_directions)
        tbl_split = tbl_split.transpose(0, 1).contiguous()
        if self.split_type == 'outcell':
            batch_index = torch.LongTensor(range(tbl_split.data.size(1))).unsqueeze_(
                0).cuda().expand_as(tbl_split.data)
            enc_split = tbl_context[tbl_split.data, batch_index, :]
            enc_left, enc_right = enc_split[:-1], enc_split[1:]
        elif self.split_type == 'incell':
            batch_index = torch.LongTensor(range(tbl_split.data.size(1))).unsqueeze_(
                0).cuda().expand(tbl_split.data.size(0) - 1, tbl_split.data.size(1))
            split_left = (tbl_split.data[:-1] +
                          1).clamp(0, tbl_context.size(0) - 1)
            enc_left = tbl_context[split_left, batch_index, :]
            split_right = (tbl_split.data[1:] -
                           1).clamp(0, tbl_context.size(0) - 1)
            enc_right = tbl_context[split_right, batch_index, :]

        if self.merge_type == 'sub':
            return (enc_right - enc_left), hidden
        elif self.merge_type == 'cat':
            # take half vector for each direction
            half_hidden_size = self.hidden_size
            return torch.cat([enc_right[:, :, :half_hidden_size], enc_left[:, :, half_hidden_size:]], 2), hidden
        elif self.merge_type == 'mlp':
            return self.merge(torch.cat([enc_right, enc_left], 2)), hidden
