import torch.utils.data as data
import glob
import os
import math
import cv2

from data.utils import *


class SRData(data.Dataset):

    def __init__(self, args, train):

        self.args = args
        self.train = train

        if self.train:
            self.dataset_path, self.lr, self.hr = self.set_file(self.args.data_path_train)
        else:
            self.dataset_path, self.lr, self.hr = self.set_file(self.args.data_path_test)

        self.lr_names, self.hr_names = self.scan()

    def set_file(self, data_path):
        return

    def scan(self):
        '''
        获取所有图片的路径
        :return: 存放 lr 和 hr 的图片的list
        '''
        lr_names = sorted(glob.glob(os.path.join(self.lr, '*')))
        hr_names = sorted(glob.glob(os.path.join(self.hr, '*')))
        assert len(lr_names) == len(hr_names)
        print('total LR/HR images:  {}'.format(len(lr_names)))

        return lr_names, hr_names

    def load_file(self, idx):
        '''

        :param idx: the idx of files
        :return: h*w*c , filename
        '''
        lr_array = cv2.imread(self.lr_names[idx])
        hr_array = cv2.imread(self.hr_names[idx])
        filename = os.path.basename(os.path.splitext(self.lr_names[idx])[0])
        return lr_array, hr_array, filename

    def __getitem__(self, idx):

        lr_array, hr_array, filename = self.load_file(idx)
        # get image patch
        lr_patch, hr_patch, hr_patch_for_gaussian = self.get_patch(lr_array, hr_array, hr_array)
        # convert to ycbcr
        lr_channel, hr_channel = set_channel(lr_patch, hr_patch)
        hr_channel_for_gaussian = set_channel(hr_patch_for_gaussian, only_y=True)[0]
        # add Gaussian Blur
        if self.args.scale == 2:
            kernel_size = int(math.ceil(0.5 * 3) * 2 + 1)
            hr_gaussian_channel = cv2.GaussianBlur(hr_channel_for_gaussian, (kernel_size, kernel_size), 0.5)
        elif self.args.scale == 3:
            kernel_size = int(math.ceil(0.9 * 3) * 2 + 1)
            hr_gaussian_channel = cv2.GaussianBlur(hr_channel_for_gaussian, (kernel_size, kernel_size), 0.9)
        elif self.args.scale == 4:
            kernel_size = int(math.ceil(1.3 * 3) * 2 + 1)
            hr_gaussian_channel = cv2.GaussianBlur(hr_channel_for_gaussian, (kernel_size, kernel_size), 1.3)

        hr_gaussian_channel = np.expand_dims(hr_gaussian_channel, axis=-1)
        # convert numpy to tensor
        hr_gaussian_tensor = np2tensor(hr_gaussian_channel, pixel_range=self.args.pixel_range)[0]
        lr_tensor, hr_tensor = np2tensor(lr_channel, hr_channel, pixel_range=self.args.pixel_range)

        return lr_tensor, hr_tensor, hr_gaussian_tensor, filename

    def __len__(self):

        return len(self.hr_names)

    def get_patch(self, lr, hr, hr_gaussian):

        if self.train:

            lr, hr, hr_gaussian = get_patch(lr, hr, hr_gaussian, patch_size=self.args.patch_size)
            if not self.args.no_augment:
                lr, hr, hr_gaussian = augment(lr, hr, hr_gaussian)

        return lr, hr, hr_gaussian
