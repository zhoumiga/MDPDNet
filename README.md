# MDPD
Crowd counting based on multi-resolution density map and parallel dilated convolution 

we propose a crowd counting network (MDPDNet) based on multi-resolution density maps and parallel dilated convolutions to reduce the influence of crowd estimation by occlusion and distortion. 

Experiments need to build a training environment

Python: 3.5.2

PyTorch: 0.4.1

downloaded dataset

[shanghaitech](https://pan.baidu.com/s/1nuAYslz)  

[ucfcc50](https://www.crcv.ucf.edu/data/ucf-cc-50/)  

[ucfQNRF](https://www.crcv.ucf.edu/data/ucf-qnrf/)  


The src file contains our code, and the core idea involves two files [models.py](https://github.com/zhoumiga/MDPD/tree/main/src) and [network.py](https://github.com/zhoumiga/MDPD/tree/main/src)。As this method is useful in one of our projects, the complete code will not be uploaded until the end of the project.However, our method draws on the framework of [CSRNet](https://github.com/leeyeehoo/CSRNet-pytorch), so friends who are interested can refer to the framework of A and the above two documents for experimental reproduction.

densitymap.py is used to generate a density map on the image test set we used.

