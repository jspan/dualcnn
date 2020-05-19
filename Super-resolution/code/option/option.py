import argparse

import template

parser = argparse.ArgumentParser(description='Dual CNN')
parser.add_argument('--template', type=str, default='Dual_CNN',
                    help='to choose which model in template, Dual_CNN or Dual_CNN-S')

# hardware config
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# data config
parser.add_argument('--data_path_train', type=str, default='.', help='train dataset directory')
parser.add_argument('--data_path_test', type=str, default='.', help='test dataset directory')
parser.add_argument('--data_train', type=str, default='.', help='train dataset name')
parser.add_argument('--data_test', type=str, default='.', help='test dataset name')
parser.add_argument('--pixel_range', type=int, default=1, help='About pixel range , set 255 or 1 ')

# training config
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training')
parser.add_argument('--patch_size', type=int, default=41, help='get the patch img')
parser.add_argument('--no_augment', action='store_true', help='do not use data augmentation')
parser.add_argument('--lr', type=float, default=1e-4, help='the initial learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
parser.add_argument('--lr_decay', type=int, default=1,
                    help='learning rate decay per N epochs')
parser.add_argument('--weight_decay', type=int, default=1e-4,
                    help='optimizer weight decay')
parser.add_argument('--clip', type=float, default=0.4, help='clip gradients,Default=0.4')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--epoch', type=int, default=500, help='training epoch')
parser.add_argument('--scale', type=int, default=2, help='')
parser.add_argument('--loss', type=str, default='L2', help='the loss function')

# network config
parser.add_argument('--model', type=str, default='.', help='choose the model')

# others
parser.add_argument('--reset', action='store_true', help='reset the training')
parser.add_argument('--save', type=str, default='.', help='folder to save result')
parser.add_argument('--print_every', type=int, default=10,
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_only', action='store_true', help='set True , do test ; else , do val')
parser.add_argument('--save_images', default=True, action='store_true', help='save_images')
parser.add_argument('--pre_train', type=str, default='.', help='pre-trained model directory')
parser.add_argument('--save_models', default=True, action='store_true', help='save all the models')
parser.add_argument('--quickly_test', default=False, action='store_true', help='quickly test')
parser.add_argument('--resume', action='store_true', help='set true to continue to train')

args = parser.parse_args()
template.set_template(args=args)
