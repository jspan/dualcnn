import os

from data import dataset


class Benchmark(dataset.SRData):

    def __init__(self, args, train=False):
        super(Benchmark, self).__init__(args, train=train)

    def scan(self):

        lr_names, hr_names = super(Benchmark, self).scan()

        return lr_names, hr_names

    def set_file(self, data_path):

        self.dataset_path = os.path.join(data_path, self.args.data_test)
        if not self.args.test_only:
            print('Start to load the validation dataset...')
            self.lr = os.path.join(self.dataset_path, 'x{}'.format(self.args.scale), 'LR')
            self.hr = os.path.join(self.dataset_path, 'x{}'.format(self.args.scale), 'HR')
            print('Validation dataset path :', self.dataset_path)
            print('Validation dataset LR path:', self.lr)
            print('Validation dataset HR path:', self.hr)
        else:
            print('Start to load the testing dataset...')
            self.lr = os.path.join(self.dataset_path, 'x{}'.format(self.args.scale), 'LR')
            self.hr = os.path.join(self.dataset_path, 'x{}'.format(self.args.scale), 'HR')
            print('Testing dataset path :', self.dataset_path)
            print('Testing dataset LR path:', self.lr)
            print('Testing dataset HR path:', self.hr)

        return self.dataset_path, self.lr, self.hr
