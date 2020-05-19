from data.utils import *
from model.dual_cnn import Dual_cnn
from option.option import args
import numpy as np
from tools.calc_image_psnr_ssim import *


def ycbcr2rgb(img):
    """same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    return rlt.astype(in_img_type)


def eval(model, test_set, save_path):
    HR_path = '../test_sets/{}/HR'.format(test_set)
    LR_path = '../test_sets/{}/LR_bicubic/X{}'.format(test_set, test_scale)
    result_path = '{}/X{}/{}'.format(save_path, test_scale, test_set)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    imgs = sorted(os.listdir(HR_path))
    with torch.no_grad():
        for img in imgs:
            LR_img = cv2.imread(os.path.join(LR_path, img))
            LR_Ycbcr = bgr2ycbcr(LR_img, only_y=False)
            LR_Y = LR_Ycbcr[:, :, 0]
            LR_cbcr = LR_Ycbcr[:, :, 1:]
            LR_in = torch.from_numpy(np.expand_dims(np.expand_dims(LR_Y, axis=0), axis=0)).float().to(device) / 255.
            _, _, SR = model(LR_in)
            SR_Y = np.expand_dims(SR[0, 0, :, :].cpu().numpy(), axis=2)
            SR_Y = SR_Y * 255.
            SR_img_Ycbcr = np.concatenate((SR_Y, LR_cbcr[6:-6, 6:-6, :]), axis=2)
            SR_img = ycbcr2rgb(SR_img_Ycbcr)
            SR_img = SR_img[:, :, ::-1]
            cv2.imwrite(os.path.join(result_path, img), SR_img)

    calc_image_PSNR_SSIM(result_path, HR_path, test_scale)


if __name__ == '__main__':
    device = torch.device('cuda')
    model = Dual_cnn(args).to(device)
    args.quickly_test = True
    test_scale = 2
    weight = torch.load('../models_in_paper/Dual_CNN/Dual_CNN_x2.pt')
    model.load_state_dict(weight, strict=False)
    test_sets = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']
    # test_sets = ['Set5']
    save_path = './test_results'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for test_set in test_sets:
        print("-------->now testing {}".format(test_set))
        eval(model, test_set, save_path)
