import decimal
import os
from tqdm import tqdm
from utils.utils import *


class Train_SR:
    def __init__(self, args, loader, model, ckp):
        self.args = args
        self.ckp = ckp
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')

        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = model
        self.optimizer = self.make_optimizer()
        self.scheduler = self.make_scheduler()

        if self.args.loss == 'L2':
            self.loss = torch.nn.MSELoss()
        elif self.args.loss == 'L1':
            self.loss = torch.nn.L1Loss()
        self.loss_dict = {}

        if self.args.resume:
            self.optimizer.load_state_dict(torch.load(os.path.join(self.ckp.dir, 'optimizer.pt')))

    def make_optimizer(self):

        return torch.optim.Adam([
            {'params': self.model.get_model().srcnn_structure.parameters()},
            {'params': self.model.get_model().vdsr_detail.parameters(), 'lr': 1e-5}

        ],
            lr=self.args.lr, weight_decay=self.args.weight_decay)

    def make_scheduler(self):
        kwargs = {'step_size': self.args.lr_decay, 'gamma': self.args.gamma}
        return torch.optim.lr_scheduler.StepLR(self.optimizer, **kwargs)

    def train(self):
        print('Image Super-resolution Training...')
        self.model.train()

        if self.args.resume:
            epoch_start = len(self.ckp.psnr_log)
        else:
            epoch_start = 0

        for epoch in range(epoch_start + 1, self.args.epoch + 1):
            self.scheduler.step(epoch=epoch)
            lr = self.scheduler.get_lr()

            self.ckp.write_log('Epoch {:3d} with SRCNN Learning rate:{:.2e}, VDSR Learning rate:{:.2e} : '.format(epoch,
                                                                                                                  decimal.Decimal(
                                                                                                                      lr[
                                                                                                                          0]),
                                                                                                                  decimal.Decimal(
                                                                                                                      lr[
                                                                                                                          -1])))
            self.ckp.start_loss_log()

            for batch, (lr, hr, hr_gaussian, filename) in enumerate(self.loader_train):

                # get Y channel
                lr = lr[:, 0:1, :, :].to(self.device)
                hr = hr[:, 0:1, 6:-6, 6:-6].to(self.device)
                hr_gaussian = hr_gaussian[:, :, 6:-6, 6:-6].to(self.device)

                self.optimizer.zero_grad()
                structure, detail, sr = self.model(lr)

                struct_loss = self.args.structure_weight * self.loss(structure, hr)
                detail_loss = self.args.detail_weight * self.loss(detail, hr - hr_gaussian)
                recon_loss = self.loss(sr, hr)
                loss = recon_loss + struct_loss + detail_loss

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.get_model().parameters(), self.args.clip)
                self.optimizer.step()

                self.loss_dict = {
                    'struct_loss': struct_loss.item(),
                    'detail_loss': detail_loss.item(),
                    'recon_loss': recon_loss.item(),
                    'loss': loss.item()
                }

                self.ckp.report_loss_log(self.loss_dict)

                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log(
                        '[{}/{}]\t[Struct Loss]={:.8f}\t[Detail Loss]={:.8f}\t[Recon Loss]={:.4f}\t[Loss]={:.4f}'.format(
                            (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                            struct_loss.item(), detail_loss.item(), recon_loss.item(), loss.item()))

            self.ckp.end_loss_log(len(self.loader_train))
            self.test()
            self.ckp.save(self, epoch=epoch)

    def test(self):
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        self.ckp.start_psnr_log()

        with torch.no_grad():
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for id, (lr, hr, hr_gaussian, filename) in enumerate(tqdm_test):

                lr_cbcr = lr[:, 1:, :, :].to(self.device)
                hr_cbcr = hr[:, 1:, :, :].to(self.device)
                lr = lr[:, 0:1, :, :].to(self.device)
                hr = hr[:, 0:1, :, :].to(self.device)

                struct, detail, sr = self.model(lr)

                lr, lr_cbcr, hr, hr_cbcr, sr = postprocess(lr, lr_cbcr, hr, hr_cbcr, sr,
                                                           rgb_range=self.args.pixel_range)  # [0 , 1]
                psnr_value = psnr(self.args, sr, hr[:, :, 6:-6, 6:-6])
                self.ckp.report_psnr_log(item=psnr_value)

                lr = torch.cat([lr, lr_cbcr], dim=1)
                hr = torch.cat([hr, hr_cbcr], dim=1)
                sr = torch.cat([sr, hr_cbcr[:, :, 6:-6, 6:-6]], dim=1)

                if self.args.save_images:
                    # save_list = [lr[:, :, 6:-6, 6:-6], hr[:, :, 6:-6, 6:-6], sr]
                    save_list = [sr]
                    self.ckp.save_images(filename, save_list)

            self.ckp.end_psnr_log(len(self.loader_test))
            best = self.ckp.psnr_log.max(0)
            self.ckp.write_log('[{}]\taverage psnr={:.3f}\t[best psnr={:.3f}\t@epoch:{}]'.format(self.args.data_test,
                                                                                                 self.ckp.psnr_log[-1],
                                                                                                 best[0], best[1] + 1))
