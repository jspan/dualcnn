# Dual CNN

This repository is an official PyTorch implementation of the paper "Dual Convolutional Neural Networks for Low-Level Vision" .  
The code is built on EDSR (PyTorch) and tested on Ubuntu 16.04 environment (Python3.6, PyTorch_0.4.1, CUDA8.0, cuDNN5.1) with NVIDIA 1080Ti GPUs.


## Dependencies
* ubuntu16.04
* Python 3.6(Recommend to use Anaconda)
* PyTorch0.4.1
* numpy
* skimage
* imageio
* matplotlib
* tqdm
* cv2 

## Super-resolution

#### Trainset:
We use an augment version of 291 natural images in the BSDS500 dataset to train our models. You can download GT from [Baidu Dirver](https://pan.baidu.com/s/1kEcL42MBPJrRZ3MXQhhjcA). (passcode: ehqd)  
And you can also download the generated LR-HR pairs from [Baidu Driver](https://pan.baidu.com/s/18IetZvOsD9xZlEIslSz6eg). (password: dtbm)

#### Benechmarks:
You can evaluate our models with widely-used benchmark datasets.

[Set5 - Bevilacqua et al. BMVC 2012](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html),

[Set14 - Zeyde et al. LNCS 2010](https://sites.google.com/site/romanzeyde/research-interests),

[B100 - Martin et al. ICCV 2001](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/),

[Urban100 - Huang et al. CVPR 2015](https://sites.google.com/site/jbhuang0604/publications/struct_sr).


#### Models
We release two versions of our model: Dual_CNN and Dual_CNN-S.   
All the models(X2, X3, X4) can be downloaded from [Baidu eDrive](https://pan.baidu.com/s/1mk1ucMGhf-ojmwvU4NwjlA). (password:i4hm)

## Quicktest with benchmark
You can test our super-resolution algorithm with benchmarks. Please organize the testset in  ``./testsets`` folder like this:  
```
|--testset  
    |--Set5  
        |--LR_bicubic
            |--X2
                |--baby.png  
                     ：   
                     ： 
            |--X3
            |--X4
        |--HR
            |--baby.png  
                 ：   
                 ： 
```

We have already made a testset with Set5, Set14, B100, Urban100 and Manga109 datasets. And you can download them from [Baidu Driver](https://pan.baidu.com/s/1CB_98H4ZAQuVFZAj1FXViA) .(password:b1so) 

After downloading models, place them in folder ``./models_in_paper``

Then, run the following commands:
```bash
cd code
python test_save.py
```
And generated results can be found in ``./test_results/``
  * To test all scales, you can modify the line50 to in test_save.py.  
  * To test all models, you can modify the line51 to in test_save.py.   
  

### How to train
If you have downloaded the trainset(GT), please use the matlab code in ``./code/matlab_tools/`` to generate LR images. 
Then, make sure that the trainset has been organized as follows:
```
|--291
    |--x2  
        |--LR
            |--xxxx.bmp  
                 ：   
                 ： 
        |--HR
            |--xxxx.bmp  
                 ：   
                 ： 
        |--x3
        |--x4 
```
And organizaton of validation set is same like that.  

We train our Dual CNN model with pretrain(SRCNN, VDSR), please download the pretrain model in [Baidu Driver](https://pan.baidu.com/s/1vh6UaidfBg-MibQhArehmg) (password:puj6) and place them in folder ``./pretrain``
The command for training is as follow:
```
cd code
python main.py --data_path_train <your trainset root> --data_path_test <your validationset root>
```
All trained models can be found in ``./code/experiment/Dual_CNN_x2/model``
  * To change the model you want to train, you can modify the option(template) in line4, ``./code/option/option.py``.  
  * To train all scales, you can modify the option(scale) in line11, ``./code/template.py``.   
  * To change the save root, you can modify the option(save) in line13, ``./code/template.py``.  