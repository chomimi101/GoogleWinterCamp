# coding: utf-8

import json
import nltk
import torch
import functools
import numpy as np
from predict.utils import Args
from torch.autograd import Variable
from predict.utils import build_vocab, change2idx, pad
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert import BertTokenizer, BertModel
from predict.utils import UNK, labels_vocab, vocab_path


class MBTI(Dataset):
    def __init__(self, path, args, data_from_train=None):
        self.args = args
        labels = []
        posts, posts_len = [], []
        if self.args.model == 'lstm':
            with open(path) as f:
                for line in f:
                    info = json.loads(line.strip())
                    labels.append(labels_vocab.get(info['label']))
                    res = info['posts'].split(' ')
                    posts.append(res), posts_len.append(len(res))

            if data_from_train is None:
                self.vocab, _ = build_vocab(posts, init_vocab={UNK: 0}, min_word_freq=5)
                self.posts_max_len = min(max(posts_len), self.args.max_len)
                print('vocab_size: {}'.format(len(self.vocab)))
                with open(vocab_path, 'w') as f:
                    f.write(json.dumps(self.vocab) + '\n')
            else:
                self.vocab, self.posts_max_len = data_from_train
            # change2tensor
            self.posts_tensor = torch.LongTensor(pad(change2idx(posts, vocab=self.vocab), max_len=self.posts_max_len))
            self.posts_len_tensor = torch.LongTensor(list(map(lambda len: min(len, self.posts_max_len), posts_len)))
            self.labels_tensor = torch.LongTensor(labels)
            # get len
            self.len = len(self.labels_tensor)
            assert len(self.labels_tensor) == len(self.posts_tensor) == len(self.posts_len_tensor)
        else:
            with open(path) as f:
                for line in f:
                    info = json.loads(line.strip())
                    labels.append(labels_vocab.get(info['label']))

    def __getitem__(self, index):
        if self.args.model == 'lstm':
            return (self.posts_tensor[index], self.posts_len_tensor[index]), self.labels_tensor[index]
        else:
            pass

    def __len__(self):
        return self.len


if __name__ == '__main__':
    pass
