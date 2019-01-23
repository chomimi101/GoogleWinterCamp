# coding: utf-8

import json
import nltk
import torch
import functools
import numpy as np
from utils import Args
from torch.autograd import Variable
from utils import build_vocab, change2idx, pad
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert import BertTokenizer, BertModel
from utils import preprocess_path, preprocess_lstm_path, UNK, labels_vocab


class MBTI(Dataset):
    def __init__(self, args, data_from_train=None):
        self.args = args
        labels = []
        posts, posts_len = [], []
        if 'lstm' in self.args.path:
            with open(self.args.path) as f:
                for line in f:
                    info = json.loads(line.strip())
                    labels.append([info['label']])
                    res = info['posts'].split(' ')
                    posts.append(res), posts_len.append(len(res))

            if data_from_train is None:
                self.vocab, _ = build_vocab(posts, init_vocab={UNK: 0}, min_word_freq=3)
                self.posts_max_len = min(max(posts_len), self.args.max_len)
            else:
                vocab, self.posts_max_len = data_from_train
            print('vocab_size: {}'.format(len(self.vocab)))

            # change2tensor
            self.posts_tensor = torch.LongTensor(pad(change2idx(posts, vocab=self.vocab), max_len=self.posts_max_len))
            self.posts_len_tensor = torch.LongTensor(list(map(lambda len: min(len, self.posts_max_len), posts_len)))
            self.labels_tensor = torch.LongTensor(change2idx(labels, vocab=labels_vocab))
            # get len
            self.len = len(self.labels_tensor)
            assert len(self.labels_tensor) == len(self.posts_tensor) == len(self.posts_len_tensor)

    def __getitem__(self, index):
        if 'lstm' in self.args.path:
            return (self.posts_tensor[index], self.posts_len_tensor[index]), self.labels_tensor[index]

    def __len__(self):
        return self.len


if __name__ == '__main__':
    pass
