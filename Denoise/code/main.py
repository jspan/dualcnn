import argparse, os
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dual_cnn_denoise import DUAL_CNN_DENOISE
from data_generator import datagenerator, DenoisingDataset
import eval


# Training settings
parser = argparse.ArgumentParser(description="PyTorch VDSR")
parser.add_argument('--data_train', default='../datasets/BSDS500/train/', type=str, help='trainset root')
parser.add_argument("--batchSize", type=int, default=8, help="Training batch size")  # todo
parser.add_argument("--nEpochs", type=int, default=500, help="Number of epochs to train for")
parser.add_argument("--srcnn_lr", type=float, default=2e-4, help="SRCNN Learning Rate. ")
parser.add_argument("--vdsr_lr", type=float, default=2e-5, help="VDSR Learning Rate. ")
parser.add_argument("--step", type=int, default=50,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=50")
parser.add_argument("--threads", type=int, default=4, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--gamma", default=0.2, type=float, help="gamma, Default: 0.9")
parser.add_argument("--weight-decay", default=1e-5, type=float,
                    help="Weight decay, Default: 1e-4")  # todo   1e-4
parser.add_argument('--save', default='./checkpoint_50/', type=str, help='path to save mdoel')
parser.add_argument("--sigma", type=int, default=50, help="noise level")  # todo


def set_logger(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logger = open('{}logger.txt'.format(save_path), 'w')
    return logger


def main():
    global opt, model
    opt = parser.parse_args()
    logger = set_logger(opt.save)
    print(opt)
    print(opt, file=logger)

    # setting gpu and seed
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    # setting dataset
    print("===> Loading dataset")
    patches = datagenerator(data_dir=opt.data_train)
    train_set = DenoisingDataset(patches, sigma=opt.sigma)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, drop_last=True,
                                      batch_size=opt.batchSize, shuffle=True)

    # setting model and loss
    print("===> Building model")
    model = DUAL_CNN_DENOISE()
    criterion = nn.MSELoss(size_average=False)
    model = model.cuda()
    criterion = criterion.cuda()

    # setting optimizer
    print("===> Setting Optimizer")
    kwargs = {'weight_decay': opt.weight_decay}
    optimizer = optim.Adam([{"params": model.structure_net.parameters(), "lr": opt.srcnn_lr},
                            {"params": model.detail_net.parameters(), "lr": opt.vdsr_lr}], **kwargs)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch, logger)
        model_path = save_checkpoint(model, epoch)
        eval.eval(model_path, opt.save, opt.sigma)


def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    srcnn_lr = opt.srcnn_lr * (opt.gamma ** (epoch // opt.step))
    vdsr_lr = opt.vdsr_lr * (opt.gamma ** (epoch // opt.step))
    return srcnn_lr, vdsr_lr


def train(training_data_loader, optimizer, model, criterion, epoch, logger):
    # setting lr_decay
    srcnn_lr, vdsr_lr = adjust_learning_rate(epoch - 1)
    optimizer.param_groups[0]["lr"] = srcnn_lr
    optimizer.param_groups[1]["lr"] = vdsr_lr

    print("Epoch = {}, srcnn_lr = {}, vdsr_lr = {}".format(epoch, optimizer.param_groups[0]["lr"],
                                                           optimizer.param_groups[1]["lr"]), file=logger)
    print("Epoch = {}, srcnn_lr = {}, vdsr_lr = {}".format(epoch, optimizer.param_groups[0]["lr"],
                                                           optimizer.param_groups[1]["lr"]))
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):

        lr_input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        lr_input = lr_input.cuda()
        target = target[:, :, 6:-6, 6:-6].cuda()

        sr_out, structure_out = model(lr_input)
        loss = criterion(sr_out, target) + 0.01 * criterion(structure_out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 3000 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader),
                                                                loss.item()), file=logger)
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader),
                                                                loss.item()))


def save_checkpoint(model, epoch):
    model_out_path = opt.save + "model_epoch_{:0>4}.pth".format(epoch)  # todo checkpoint
    state = {"epoch": epoch, "model": model}
    torch.save(state, model_out_path)
    return model_out_path


if __name__ == "__main__":
    main()
