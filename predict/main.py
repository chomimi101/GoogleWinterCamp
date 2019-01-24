# coding: utf-8

import torch
from torch import nn
from predict.models.baseline import Baseline
from predict.utils import Args, preprocess_lstm_train, preprocess_lstm_test
from predict.dataloader import MBTI
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score


class Main(object):
    def __init__(self, args):
        self.args = args

    def train(self):
        # build train_dataloader
        train_dataset = MBTI(path=preprocess_lstm_train, args=self.args)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=self.args.shuffle)
        data_from_train = train_dataset.vocab, train_dataset.posts_max_len
        self.args.vocab, self.args.posts_max_len = data_from_train
        print(self.args.posts_max_len)
        self.args.vocab_size = len(self.args.vocab)
        # build test_dataloader
        test_dataset = MBTI(path=preprocess_lstm_test, args=self.args, data_from_train=data_from_train)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.args.batch_size, shuffle=self.args.shuffle)
        # train
        model = Baseline(self.args)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        model.to(self.args.device)
        for epoch in range(self.args.epoch):
            model.train()
            epoch_loss = 0
            for data in train_dataloader:
                model.zero_grad(), optimizer.zero_grad()
                inputs, labels = data
                logits = model(inputs)
                labels = labels.to(self.args.device)
                # print(logits)
                # print('$$$$$$$')
                # print(torch.max(logits, -1)[1].data.cpu().numpy().tolist())
                # print('#######')
                # print(labels)
                loss = criterion(logits, labels)
                epoch_loss += loss.data
                loss.backward()
                optimizer.step()
            self.test(test_dataloader, model, epoch)
            print('epoch {} loss {}'.format(epoch, epoch_loss))
            print('train')
            self.test(train_dataloader, model)
        torch.save(model, './res/lstm_torch.model')

    def test(self, dataloader, model, epoch=None):
        model.eval()
        total_true, total_pred = [], []
        for data in dataloader:
            inputs, labels = data
            logits = model(inputs)
            pred = torch.max(logits, -1)[1].data.cpu().numpy().tolist()
            true = labels.data.cpu().numpy().tolist()
            total_true.extend(true), total_pred.extend(pred)
        if epoch is not None:
            print('epoch {}'.format(epoch))
        print('micro f1: {}'.format(f1_score(total_true, total_pred, average='micro')))
        print(classification_report(total_true, total_pred))
        print(confusion_matrix(total_true, total_pred))
        print(accuracy_score(total_true, total_pred))


if __name__ == '__main__':
    args = Args()
    main = Main(args)
    main.train()
