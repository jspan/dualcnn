import os

from data import dataset


class B500(dataset.SRData):

    def __init__(self, args, train=True):
        super(B500, self).__init__(args, train=train)

    def scan(self):
        lr_names, hr_names = super(B500, self).scan()

        return lr_names, hr_names

    def set_file(self, data_path):
        print('Start to load the training dataset...')
        self.dataset_path = os.path.join(data_path, self.args.data_train)
        self.lr = os.path.join(self.dataset_path, 'x{}'.format(self.args.scale), 'LR')
        self.hr = os.path.join(self.dataset_path, 'x{}'.format(self.args.scale), 'HR')
        print('Training dataset path:', self.dataset_path)
        print('Training dataset LR path:', self.lr)
        print('Training dataset HR path:', self.hr)
        return self.dataset_path, self.lr, self.hr
