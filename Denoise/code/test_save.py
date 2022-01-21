import torch
from torch.autograd import Variable
import numpy as np
import math
from tools.calc_image_psnr_ssim import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def eval(test_model, sigma, gt_path, result_path):
    image_list = os.listdir(gt_path)

    model = torch.load(test_model)["model"]
    count = 0
    for image_name in image_list:
        with torch.no_grad():
            im_gt_y = cv2.imread(os.path.join(gt_path, image_name), 0)
            im_gt_y = im_gt_y.astype(float)
            im_gt_y = im_gt_y / 255.
            torch.manual_seed(0)
            noise = torch.randn(im_gt_y.shape) * (sigma / 255.0)
            im_input = im_gt_y + noise
            im_input = im_input.numpy()

            im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

            model = model.cuda()
            im_input = im_input.cuda()

            HR, _ = model(im_input)
            HR = HR.cpu()
            im_h_y = HR.data[0].numpy().astype(np.float32)

            im_h_y = im_h_y * 255.
            im_h_y[im_h_y < 0] = 0
            im_h_y[im_h_y > 255.] = 255.
            im_h_y = im_h_y[0, :, :]
            cv2.imwrite(os.path.join(result_path, '{}.png'.format(image_name.split('.')[0])), im_h_y)
            count += 1
            print("{} images have proceeded!".format(count))
    calc_image_PSNR_SSIM(result_path, gt_path)


if __name__ == '__main__':
    sigma = 15
    gt_path = "../dataset/BSDS500/test"
    result_path = './test_results/sigma_{}'.format(sigma)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    model_path = "../models_in_paper/sigma{}.pth".format(sigma)
    eval(model_path, sigma, gt_path, result_path)
