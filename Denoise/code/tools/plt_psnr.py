import numpy as np
from matplotlib import pyplot as plt


def plot_psnr_log(psnr_list):
    epoch = len(psnr_list)
    axis = np.linspace(1, epoch, epoch)
    fig = plt.figure()
    plt.title('PSNR Graph')
    plt.plot(axis, np.array(psnr_list))
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.savefig('psnr_NOISE15.pdf')
    plt.close(fig)


if __name__ == '__main__':
    f = open('./report_denoise_bsd68_NOISE15.txt', 'r')
    lines = f.readlines()
    psnr_list = []
    for line in lines:
        psnr = float(line.split('=')[1][:-1])
        psnr_list.append(psnr)
    plot_psnr_log(psnr_list)
