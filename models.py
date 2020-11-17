
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.network import Conv2d


# from torchvision import models


#MDPD
class MDPD(nn.Module):
    def __init__(self, bn=False):
        super(MDPD, self).__init__()

        self.conv_pre = nn.Sequential(Conv2d(3, 64, 3, same_padding=True, bn=bn),
                                      Conv2d(64, 64, 3, same_padding=True, bn=bn),
                                      nn.MaxPool2d(2),
                                      Conv2d(64, 128, 3, same_padding=True, bn=bn),
                                      Conv2d(128, 128, 3, same_padding=True, bn=bn),
                                      nn.MaxPool2d(2))
        self.conv_end = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                      Conv2d(128, 1, 1, same_padding=True, bn=bn))

        # branch 1 convolution to extract features
        self.branch1 = nn.Sequential(Conv2d(128, 256, 3, same_padding=True, bn=bn),
                                     Conv2d(256, 256, 3, same_padding=True, bn=bn),
                                     Conv2d(256, 256, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2))
        # generating the density map, upsampling becomes a feature map
        self.fu1 = nn.Sequential(Conv2d(256, 1, 1, same_padding=True, bn=bn),
                                 nn.Upsample(scale_factor=2, mode='nearest'),
                                 Conv2d(1, 64, 3, same_padding=True, bn=bn),
                                 nn.Upsample(scale_factor=2, mode='nearest'))  # 64
        # branch 2
        self.branch2 = nn.Sequential(Conv2d(256, 512, 3, same_padding=True, bn=bn),
                                     Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                     Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2))
        self.fu2 = nn.Sequential(Conv2d(512, 1, 1, same_padding=True, bn=bn),
                                 nn.Upsample(scale_factor=2, mode='nearest'),
                                 Conv2d(1, 64, 3, same_padding=True, bn=bn),
                                 nn.Upsample(scale_factor=4, mode='nearest'))
        # branch 3
        self.branch3 = nn.Sequential(Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                     Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                     Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2))
        self.fu3 = nn.Sequential(Conv2d(512, 1, 1, same_padding=True, bn=bn),
                                 nn.Upsample(scale_factor=4, mode='nearest'),
                                 Conv2d(1, 64, 3, same_padding=True, bn=bn),
                                 nn.Upsample(scale_factor=4, mode='nearest'))
        self.branch = nn.Sequential(Conv2d(64, 128, 3, same_padding=True, bn=bn))
        '''
        # 分支1 卷积提取特征
        self.branch1 = nn.Sequential(Conv2d(1, 16, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2))
        # 生成密度图之后再上采样成为特征图
        self.fu1 = nn.Sequential(Conv2d(16, 1, 1, same_padding=True, bn=bn),)
                                 # nn.Upsample(scale_factor=2, mode='nearest'))  # 64张量
        # 分支2
        self.branch2 = nn.Sequential(Conv2d(16, 32, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2))
        self.fu2 = nn.Sequential(Conv2d(32, 1, 1, same_padding=True, bn=bn),)
                                 # nn.Upsample(scale_factor=2, mode='nearest'))
        # 分支3
        self.branch3 = nn.Sequential(Conv2d(32, 64, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2))
        self.fu3 = nn.Sequential(Conv2d(64, 1, 1, same_padding=True, bn=bn),)
                                 # nn.Upsample(scale_factor=2, mode='nearest'))

        self.fuse = Conv2d(3, 1, 1, same_padding=True, bn=bn)  # 生成一张低密度图
        self.pool = nn.MaxPool2d(4)
        '''

        # dilated conv
        self.d_conv_pre = nn.Sequential(Conv2d(128, 256, 7, same_padding=True, bn=bn),
                                        nn.ReLU(),
                                        Conv2d(256, 512, 5, same_padding=True, bn=bn),
                                        nn.ReLU())
        self.d_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1))
        self.d_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2))
        self.d_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=5, dilation=5))

        self.d_conv_end = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                        Conv2d(256, 128, 3, same_padding=True, bn=bn),
                                        nn.ReLU())

    def forward(self, im_data):

        x = self.conv_pre(im_data)
        # multi-resolution density map
        x_br1 = self.branch1(x)
        x_br1_out = self.fu1(x_br1)
        x_br2 = self.branch2(x_br1)
        x_br2_out = self.fu1(x_br2)
        x_br3 = self.branch3(x_br2)
        x_br3_out = self.fu1(x_br3)
        x_branch = torch.cat((x_br1_out, x_br2_out, x_br3_out), 1)  # dim=0 splicing by row， dim=1 splicing by col
        x_branch_out = self.branch(x_branch)

        # dilated conv
        x_d_in = self.d_conv_pre(x)
        x_d1 = self.d_conv1(x_d_in)
        x_d2 = self.d_conv2(x_d_in)
        x_d3 = self.d_conv3(x_d_in)
        x_d_conv = torch.cat((x_d1, x_d2, x_d3), 1)  # dim=0 splicing by row， dim=1 splicing by col
        x_d_conv_out = self.d_conv_end(x_d_conv)

        # fusion
        x_fuse = torch.cat((x_branch_out, x_d_conv_out), 1)  # dim=0 splicing by row， dim=1 splicing by col
        x_out = self.conv_end(x_fuse)

        return x_out

