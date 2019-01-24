# coding: utf-8

import re
import json
import torch
import pandas as pd
from torch import nn
from pytorch_pretrained_bert import BertTokenizer

bert_model = 'bert-base-uncased'
data_path = './data/mbti_1.csv'
vocab_path = './data/vocab'
preprocess_train = './data/preprocess_mbti_train.csv'
preprocess_test = './data/preprocess_mbti_test.csv'
preprocess_lstm_train = './data/preprocess_lstm_mbti_train.csv'
preprocess_lstm_test = './data/preprocess_lstm_mbti_test.csv'

UNK = '<UNK>'
SEP = '<SEP>'
labels_vocab = {'ENFJ': 0, 'ENFP': 1, 'ENTJ': 2, 'ENTP': 3, 'ESFJ': 4, 'ESFP': 5, 'ESTJ': 6, 'ESTP': 7, 'INFJ': 8,
                'INFP': 9, 'INTJ': 10, 'INTP': 11, 'ISFJ': 12, 'ISFP': 13, 'ISTJ': 14, 'ISTP': 15}


class Args(object):
    def __init__(self):
        self.lr = 1e-3
        self.small_lr = 1e-5
        self.weight_decay = 1e-4
        self.train_test_ratio = 0.9
        self.shuffle = True
        self.model = 'lstm'
        self.batch_size = 64
        self.epoch = 10
        self.hidden_size = 100
        self.word_dim = 100
        self.num_layers = 1
        self.dropout_p = 0.1
        self.embed_matrix = None
        self.max_len = 500
        self.class_num = 16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to clean data
def post_cleaner(post):
    # Covert all uppercase characters to lower case
    post = post.lower()
    # Remove |||
    # post = post.replace('|||', "")
    # Remove URLs, links etc
    post = re.sub(
        r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
        '', post, flags=re.MULTILINE)
    # This would have removed most of the links but probably not all
    # Remove puntuations
    puncs1 = {'@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '+', '=', '{', '}', '[', ']', '|', '\\',
              '<', '>', '/', ',', '\'', '"', ';', ':', '...'}
    for label in labels_vocab.keys():
        puncs1.add(label)
    for punc in puncs1:
        post = post.replace(punc, ' ')
    puncs2 = {'.', '?', '!', '\n'}
    for punc in puncs2:
        post = post.replace(punc, ' ' + SEP + ' ')
    # Remove extra white spaces
    post = re.sub('\s+', ' ', post).strip()
    return post


def preprocess(data_path, out_path1, out_path2, ratio):
    text = pd.read_csv(data_path)
    train_num = ratio * text.shape[0]
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    total_len, count = 0, 0
    with open(out_path1, 'w') as out_f1, open(out_path2, 'w') as out_f2:
        for i in range(text.shape[0]):
            info = text.iloc[i]
            preprocess_info = {}
            preprocess_info['label'] = info['type']
            preprocess_info['posts_list'] = [post for post in post_cleaner(info['posts']).split('|||') if len(post) != 0]
            preprocess_info['bert_list'] = []
            preprocess_info['bert_index_list'] = []
            for index, post in enumerate(preprocess_info['posts_list']):
                sents = post.split(SEP)
                for sent in sents:
                    tokenize_res = tokenizer.tokenize(sent)
                    total_len += len(tokenize_res)
                    if len(tokenize_res) != 0:
                        count += 1
                        preprocess_info['bert_list'].append(tokenize_res)
                        preprocess_info['bert_index_list'].append(tokenizer.convert_tokens_to_ids(tokenize_res))
            if i < train_num:
                out_f1.write(json.dumps(preprocess_info) + '\n')
            else:
                out_f2.write(json.dumps(preprocess_info) + '\n')
    print(count, total_len / count)


def preprocess_lstm(data_path, out_path1, out_path2, ratio):
    text = pd.read_csv(data_path)
    train_num = ratio * text.shape[0]
    with open(out_path1, 'w') as out_f1, open(out_path2, 'w') as out_f2:
        for i in range(text.shape[0]):
            info = text.iloc[i]
            preprocess_info = {}
            preprocess_info['label'] = info['type']
            post = post_cleaner(info['posts']).replace('|||', ' ')
            post = re.sub('\s+', ' ', post).strip()
            preprocess_info['posts'] = post
            if i < train_num:
                out_f1.write(json.dumps(preprocess_info) + '\n')
            else:
                out_f2.write(json.dumps(preprocess_info) + '\n')


def sequence_mask(lengths, max_len=None, device=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len).to(lengths.device).type_as(lengths).repeat(batch_size, 1).lt(lengths.unsqueeze(1))


def change2idx(m_lists, vocab, oov_token=0, name='change2idx'):
    idxs_list = []
    oov_count, total = 0, 0
    for s_list in m_lists:
        # change2idx
        idxs = []
        for word in s_list:
            if word not in vocab:
                oov_count += 1
            total += 1
            idxs.append(vocab.get(word, oov_token))
        idxs_list.append(idxs)
    print('{}: oov_count - {}, total_count - {}'.format(name, oov_count, total))
    return idxs_list


def pad(m_lists, max_len, pad_token=0):
    idxs_list = []
    for s_list in m_lists:
        if len(s_list) < max_len:
            pad = [pad_token for _ in range(max_len - len(s_list))]
            s_list.extend(pad)
        else:
            s_list = s_list[:max_len]
        idxs_list.append(s_list)
    return idxs_list


def build_vocab(m_lists, pre_func=None, init_vocab=None, sort=True, min_word_freq=1):
    """
    :param m_lists: short for many lists, means list of list.
    :param pre_func: preprocess function for every word in a single list.
    :param init_vocab: init_vocab.
    :param min_count: min_count.
    :return: word2index and index2word.
    """
    # get word count
    word_count = {}
    for s_list in m_lists:
        for word in s_list:
            if pre_func is not None:
                word = pre_func(word)
            word_count[word] = word_count.get(word, 0) + 1
    # filter rare words
    new_word_count_keys = [key for key in word_count if word_count[key] >= min_word_freq]
    # sort
    if sort:
        new_word_count_keys = sorted(new_word_count_keys, key=lambda x: word_count[x], reverse=True)
    # init
    index2word = {}
    if init_vocab is None:
        word2index = {}
        num = 0
    else:
        word2index = init_vocab
        num = len(init_vocab)
        for k, v in word2index.items():
            index2word[v] = k
    # get word2index and index2word
    word2index.update(dict(list(zip(new_word_count_keys, list(range(num, num + len(new_word_count_keys)))))))
    index2word.update(dict(list(zip(list(range(num, num + len(new_word_count_keys))), new_word_count_keys))))
    return word2index, index2word


def runBiRNN(rnn, inputs, seq_lengths, hidden=None, total_length=None):
    """
    :param rnn: RNN instance
    :param inputs: FloatTensor, shape [batch, time, dim] if rnn.batch_first else [time, batch, dim]
    :param seq_lengths: LongTensor shape [batch]
    :return: the result of rnn layer,
    """
    batch_first = rnn.batch_first
    # assume seq_lengths = [3, 5, 2]
    # 对序列长度进行排序(降序), sorted_seq_lengths = [5, 3, 2]
    # indices 为 [1, 0, 2], indices 的值可以这么用语言表述
    # 原来 batch 中在 0 位置的值, 现在在位置 1 上.
    # 原来 batch 中在 1 位置的值, 现在在位置 0 上.
    # 原来 batch 中在 2 位置的值, 现在在位置 2 上.
    sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=True)

    # 如果我们想要将计算的结果恢复排序前的顺序的话,
    # 只需要对 indices 再次排序(升序),会得到 [0, 1, 2],
    # desorted_indices 的结果就是 [1, 0, 2]
    # 使用 desorted_indices 对计算结果进行索引就可以了.
    _, desorted_indices = torch.sort(indices, descending=False)

    # 对原始序列进行排序
    if batch_first:
        inputs = inputs[indices]
    else:
        inputs = inputs[:, indices]
    packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs,
                                                      sorted_seq_lengths.cpu().numpy(),
                                                      batch_first=batch_first)

    res, hidden = rnn(packed_inputs, hidden)

    padded_res, _ = nn.utils.rnn.pad_packed_sequence(res, batch_first=batch_first, total_length=total_length)
    # 恢复排序前的样本顺序
    if batch_first:
        desorted_res = padded_res[desorted_indices]
    else:
        desorted_res = padded_res[:, desorted_indices]

    if isinstance(hidden, tuple):
        hidden = list(hidden)
        hidden[0] = hidden[0][:, desorted_indices]
        hidden[1] = hidden[1][:, desorted_indices]
    else:
        hidden = hidden[:, desorted_indices]

    return desorted_res, hidden


def fix_hidden(h):
    """
    The encoder hidden is  (layers*directions) x batch x dim.
    We need to convert it to layers x batch x (directions*dim).
    """
    h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
    return h


if __name__ == '__main__':
    args = Args()
    preprocess(data_path, preprocess_train, preprocess_test, args.train_test_ratio)
    # preprocess_lstm(data_path, preprocess_lstm_train, preprocess_lstm_test, args.train_test_ratio)
    # pass
