# coding: utf-8

from utils import Args
from dataloader import MBTI
from torch.utils.data import Dataset, DataLoader


class Main(object):
    def __init__(self, args):
        self.args = args

    def train(self):
        # build train_dataloader, 136 batch
        train_dataset = MBTI(args=args)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=self.args.shuffle)
        for data in train_dataloader:
            print(data)
            logit = model(inputs)
            break

    def test(self):
        pass


if __name__ == '__main__':
    args = Args()
    main = Main(args)
    main.train()
    main.test()
