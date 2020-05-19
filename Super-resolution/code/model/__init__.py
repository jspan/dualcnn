import torch.nn as nn
import torch
from importlib import import_module
import os


class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('making model ...')

        self.args = args
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.ckp = ckp

        module = import_module('model.' + self.args.model)
        self.model = module.make_model(self.args).to(self.device)
        if not self.args.cpu and self.args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(self.args.n_GPUs))  # 多卡train

        self.load(
            apath=self.ckp.dir,
            pre_train=self.args.pre_train,
            resume=self.args.resume,
            cpu=self.args.cpu
        )

        print(self.get_model(), file=self.ckp.log_file)  # model写到log.txt文件中

    def forward(self, args):

        return self.model(args)

    def get_model(self):
        if self.args.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch):
        target = self.get_model()
        file_name = 'model_{}'.format(epoch)
        torch.save(
            target.state_dict(),
            os.path.join(apath, 'model', 'model_latest.pt')
        )

        if self.args.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', '{}.pt'.format(file_name))
            )

    def load(self, apath, pre_train='.', resume=False, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if pre_train != '.':
            print('Loading model from {}'.format(pre_train))
            print()
            self.get_model().load_state_dict(
                torch.load(pre_train, **kwargs),
                strict=False
            )
            print()
        elif resume:
            print('Loading model from {}'.format(os.path.join(apath, 'model', 'model_latest.pt')))
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_latest.pt'),
                    **kwargs
                ),
                strict=False
            )
