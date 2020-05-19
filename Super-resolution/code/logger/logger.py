import os
import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, args):

        self.args = args
        self.loss_log = {
            'struct_loss_log': torch.Tensor(),
            'detail_loss_log': torch.Tensor(),
            'recon_loss_log': torch.Tensor(),
            'loss_log': torch.Tensor()
        }
        self.psnr_log = torch.Tensor()

        self.dir = 'experiment/' + self.args.save

        if self.args.reset:
            os.system('rm -rf {}'.format(self.dir))

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            if not os.path.exists(self.dir + '/model'):
                os.makedirs(self.dir + '/model')

        if self.args.resume:
            self.psnr_log = torch.load(self.dir + '/psnr_log.pt')
            self.loss_log = torch.load(self.dir + '/loss_log.pt')
            print('Continue from epoch {}...'.format(len(self.psnr_log)))

        print('The path which results are saved : {}'.format(self.dir))

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write('From epoch {}...'.format(len(self.psnr_log)) + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def write_log(self, log):
        print(log)
        self.log_file.write(log + '\n')

    def start_loss_log(self):

        self.loss_log['struct_loss_log'] = torch.cat((self.loss_log['struct_loss_log'], torch.zeros(1)))
        self.loss_log['detail_loss_log'] = torch.cat((self.loss_log['detail_loss_log'], torch.zeros(1)))
        self.loss_log['recon_loss_log'] = torch.cat((self.loss_log['recon_loss_log'], torch.zeros(1)))
        self.loss_log['loss_log'] = torch.cat((self.loss_log['loss_log'], torch.zeros(1)))

    def report_loss_log(self, loss_dict):

        self.loss_log['struct_loss_log'][-1] += loss_dict['struct_loss']
        self.loss_log['detail_loss_log'][-1] += loss_dict['detail_loss']
        self.loss_log['recon_loss_log'][-1] += loss_dict['recon_loss']
        self.loss_log['loss_log'][-1] += loss_dict['loss']

    def end_loss_log(self, n_div):

        self.loss_log['struct_loss_log'][-1].div_(n_div)
        self.loss_log['detail_loss_log'][-1].div_(n_div)
        self.loss_log['recon_loss_log'][-1].div_(n_div)
        self.loss_log['loss_log'][-1].div_(n_div)

    def start_psnr_log(self):

        self.psnr_log = torch.cat((self.psnr_log, torch.zeros(1)))

    def report_psnr_log(self, item):
        self.psnr_log[-1] += item

    def end_psnr_log(self, n_div):

        self.psnr_log[-1].div_(n_div)

    def ycbcr2rgb(self, ycbcr_img):
        mat = np.array(
            [[65.481, 128.553, 24.966],
             [-37.797, -74.203, 112.0],
             [112.0, -93.786, -18.214]])
        mat_inv = np.linalg.inv(mat)
        offset = np.array([16, 128, 128])
        rgb_img = np.zeros(ycbcr_img.shape, dtype=np.uint8)
        for x in range(ycbcr_img.shape[0]):
            for y in range(ycbcr_img.shape[1]):
                [r, g, b] = ycbcr_img[x, y, :]
                rgb_img[x, y, :] = np.maximum(0, np.minimum(255, np.round(
                    np.dot(mat_inv, ycbcr_img[x, y, :] - offset) * 255.0)))
        return rgb_img

    def save_images(self, filename, save_list):

        store_path = '{}/result/{}'.format(self.dir, self.args.data_test)

        if not os.path.exists(store_path):
            os.makedirs(store_path)
        # postfix = ['lr', 'hr', 'sr']
        postfix = ['sr']

        for img, post in zip(save_list, postfix):
            img = img[0].data.mul(255 / self.args.pixel_range)
            img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype('uint8')
            img = self.ycbcr2rgb(img)
            imageio.imwrite(os.path.join(store_path, '{}_{}.png'.format(filename[0], post)), img)

    def save(self, trainer, epoch):

        trainer.model.save(self.dir, epoch=epoch)
        torch.save(trainer.optimizer.state_dict(), os.path.join(self.dir, 'optimizer.pt'))
        torch.save(self.loss_log, os.path.join(self.dir, 'loss_log.pt'))
        torch.save(self.psnr_log, os.path.join(self.dir, 'psnr_log.pt'))
        self.plot_loss_log(self.loss_log, epoch=epoch)
        self.plot_psnr_log(self.psnr_log, epoch=epoch)

    def plot_loss_log(self, loss_dict, epoch):

        axis = np.linspace(1, epoch, epoch)
        for loss_name, loss_value in loss_dict.items():
            fig = plt.figure()
            plt.title(loss_name.upper() + ' Graph')
            plt.plot(axis, loss_value.numpy())
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.grid(True)
            plt.savefig(os.path.join(self.dir, loss_name + '.pdf'))
            plt.close(fig)

    def plot_psnr_log(self, psnr_log, epoch):

        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title('PSNR Graph')
        plt.plot(axis, psnr_log.numpy())
        plt.xlabel('epochs')
        plt.ylabel('psnr')
        plt.grid(True)
        plt.savefig(os.path.join(self.dir, 'psnr.pdf'))
        plt.close(fig)

    def done(self):
        self.log_file.close()
