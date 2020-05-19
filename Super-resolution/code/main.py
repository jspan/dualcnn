import data
from option.option import args
from model import *
from trainer.train_sr import *
from logger.logger import *

torch.manual_seed(args.seed)
ckp = Logger(args=args)

loader = data.Data(args=args)
model = Model(args=args, ckp=ckp)
if args.model == 'dual_cnn':
    if not args.test_only:
        Trainer = Train_SR(args=args, loader=loader, model=model, ckp=ckp).train()
    else:
        Tester = Train_SR(args=args, loader=loader, model=model, ckp=ckp).test()

ckp.done()
