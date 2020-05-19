## Denoise

#### Trainset:
We use BSDS500 dataset to train our denoise models. You can download it from [Baidu Dirver](https://pan.baidu.com/s/1NPacibQm1rMV6CpP1YaZ6A) (passcode:wau9) 

#### Models
We release our model of three levels of sigma(15, 25, 50).   
All the models(sigma15, sigma25, sigma50) can be downloaded from [Baidu Drive](https://pan.baidu.com/s/1sumwg-bg-teSwEOxWbNQGg) (passcode:c66c).

## Quicktest with benchmark
You can test our super-resolution algorithm with BSDS500 testset. And you can download it from [Baidu Disk](https://pan.baidu.com/s/1OTFuML0kjJFxOIkbeU1_bw) (passcode:asv5).   
```
|--datasets  
    |--BSDS500  
        |--test
            |--X2
                |--2018.jpg  
                |--3096.jpg  
                     ：   
                     ： 
```
After downloading models, place them in folder ``./models_in_paper``

Then, run the following commands:
```bash
cd code
python test_save.py
```
And generated results can be found in ``./code/test_results/``
  * To test all sigmas, you can modify the line56 to in test_save.py.   
  

### How to train
Please downloaded the trainset first. And place trainset in ``./datasets/``
Then, make sure that the trainset has been organized as follows:
```
|--datasets
    |--BSDS500  
        |--train
            |--2092.jpg
            |--8049.jpg
                 ：   
                 ： 
```
We use CBSD68 as our validation set. You can download it from [Baidu Disk](https://pan.baidu.com/s/1xinhwu8z4zZU45Y7MDDRLQ) (passcode:kjw5). And make sure organizaton of validation set is like this:  
```
|--datasets
    |--CBSD68  
        |--0000.png
        |--0001.png
             ： 
```


We train our Dual CNN model with pretrain(SRCNN, VDSR), please download the pretrain model in [Baidu Driver](https://pan.baidu.com/s/1xP_A4XofJ_Du0rWEA67Qng) (passcode:1abl) and place them in folder ``./pretrain``
The command for training is as follow:
```
cd code
python main.py 
```
All trained models can be found in ``./code/checkpoint_xx``
  * To change the sigma you want to train, you can modify the option(sigma) in line27, ``./code/main.py``.  
  * To change the save root, you can modify the option(save) in line26, ``./code/main.py``.  