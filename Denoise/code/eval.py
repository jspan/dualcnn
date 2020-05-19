import torch
from torch.autograd import Variable
import numpy as np
import time, math
import cv2
import os


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def eval(test_model, save_path, sigma):
    gt_path = "../datasets/CBSD68"
    image_list = os.listdir(gt_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    model = torch.load(test_model)["model"]
    avg_psnr_predicted = 0.0
    avg_elapsed_time = 0.0
    count = 0.0
    for image_name in image_list:
        count += 1
        im_gt_y = cv2.imread(os.path.join(gt_path, image_name), 0)
        im_gt_y = im_gt_y.astype(float)
        im_gt_y = im_gt_y / 255.
        noise = torch.randn(im_gt_y.shape) * (sigma / 255.0)
        im_input = im_gt_y + noise
        im_input = im_input.numpy()

        im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

        model = model.cuda()
        im_input = im_input.cuda()

        start_time = time.time()
        HR, _ = model(im_input)
        elapsed_time = time.time() - start_time
        avg_elapsed_time += elapsed_time

        HR = HR.cpu()

        im_h_y = HR.data[0].numpy().astype(np.float32)

        im_h_y = im_h_y * 255.
        im_h_y[im_h_y < 0] = 0
        im_h_y[im_h_y > 255.] = 255.
        im_h_y = im_h_y[0, :, :]

        psnr_predicted = PSNR(im_gt_y[6:-6, 6:-6] * 255., im_h_y, shave_border=0)
        avg_psnr_predicted += psnr_predicted

    epoch = os.path.splitext(os.path.basename(test_model))[0]
    print('test {}'.format(epoch))
    print("PSNR_predicted=", avg_psnr_predicted / count)
    file = open(r'{}report_denoise.txt'.format(save_path), mode='a')
    file.write('\n')
    file.write(epoch)
    file.write('\t')
    file.write("PSNR_predicted={}".format(avg_psnr_predicted / count))
