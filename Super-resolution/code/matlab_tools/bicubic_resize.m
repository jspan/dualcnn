clear;close all;

%folder = '/home/weihao/weihao/sr_dataset/Set5/train/';
HRpath= './test_sets/Manga109/HR/';

% lr_save = '/home/jinshan/data3/Dual-CNN/Dual_CNN_test/dual_CNN_test/test_sets/Manga109/x2/LR/';
gt_save = '/home/jinshan/data3/Dual-CNN/Dual_CNN_test/dual_CNN_test/test_sets/Manga109/x2/HR/';
bic_save = '/home/jinshan/data3/Dual-CNN/Dual_CNN_test/dual_CNN_test/test_sets/Manga109/x2/LR_bicubic/';
scale = 2;  

filepaths = [];
filepaths = [filepaths; dir(fullfile(HRpath, '*.bmp'))];

for i = 1 : length(filepaths) 
% for i = 1 : 800
    image = imread(fullfile(HRpath,filepaths(i).name));
    
    image_gt = modcrop(image, scale);    
    imwrite(image_gt, [gt_save,filepaths(i).name]);
    
    [hei,wid,channel] = size(image);
    image_lr = imresize(image_gt,1/scale,'bicubic');
    % imwrite(image_lr, [lr_save,filepaths(i).name]);
    
    image_lr_bic = imresize(image_lr,scale,'bicubic');
    imwrite(image_lr_bic, [bic_save,filepaths(i).name]);
    
    filepaths(i).name
    % imwrite(image_gt, [gt_save,filepaths(i).name]);
end