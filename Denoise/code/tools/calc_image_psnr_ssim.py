import os
import cv2
from tools import data_utils


def calc_image_PSNR_SSIM(ouput_root, gt_root):
    PSNR_list = []
    SSIM_list = []
    output_img_list = sorted(os.listdir(ouput_root))
    gt_img_list = sorted(os.listdir(gt_root))
    for o_im, g_im in zip(output_img_list, gt_img_list):
        o_im_path = os.path.join(ouput_root, o_im)
        g_im_path = os.path.join(gt_root, g_im)

        im_GT = cv2.imread(g_im_path, 0) / 255.
        im_Gen = cv2.imread(o_im_path, 0) / 255.
        im_GT = im_GT[6:-6, 6:-6]

        assert im_GT.shape == im_Gen.shape

        # crop borders
        if im_GT.ndim == 3 or im_GT.ndim == 2:
            cropped_GT = im_GT
            cropped_Gen = im_Gen
        else:
            raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT.ndim))

        psnr = data_utils.PSNR_EDVR(cropped_GT * 255, cropped_Gen * 255)
        ssim = data_utils.SSIM_EDVR(cropped_GT * 255, cropped_Gen * 255)
        PSNR_list.append(psnr)
        SSIM_list.append(ssim)

    print('Average PSNR/SSIM={:.5}/{:.4}'.format(sum(PSNR_list) / len(PSNR_list), sum(SSIM_list) / len(SSIM_list)))

    return PSNR_list, SSIM_list
