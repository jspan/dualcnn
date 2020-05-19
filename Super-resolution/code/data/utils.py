import random
import numpy as np
import torch
import skimage.color as sc


def get_patch(*args, patch_size):
    img_h, img_w, c = args[0].shape

    ih = random.randrange(0, img_h - patch_size + 1)
    iw = random.randrange(0, img_w - patch_size + 1)

    ret = []

    for arg in args:
        ret.append(arg[ih:ih + patch_size, iw:iw + patch_size, :])

    return ret


def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = np.rot90(img)

        return img

    return [_augment(arg) for arg in args]


def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def set_channel(*l, only_y=False):
    def _set_channel(img):
        return bgr2ycbcr(img, only_y=only_y)

    return [_set_channel(_l) for _l in l]


def np2tensor(*args, pixel_range):
    def _np2tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(pixel_range / 255)

        return tensor

    return [_np2tensor(arg) for arg in args]
