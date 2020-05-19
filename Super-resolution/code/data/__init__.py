from importlib import import_module
from torch.utils.data import DataLoader

class Data():

    def __init__(self , args):
        self.args = args

        if not self.args.test_only:

            m_train = import_module('data.b500')
            train_set = getattr(m_train, 'B500')(self.args)
            self.loader_train = DataLoader(train_set,
                                       batch_size=self.args.batch_size,
                                       shuffle=True,
                                       pin_memory=not self.args.cpu,
                                       num_workers=self.args.n_threads)
        else:
            self.loader_train = None

        m_test = import_module('data.benchmark')
        test_set = getattr(m_test, 'Benchmark')(self.args)
        self.loader_test = DataLoader(test_set,
                                      batch_size=1,
                                      shuffle=False,
                                      pin_memory=not self.args.cpu,
                                      num_workers=self.args.n_threads)


