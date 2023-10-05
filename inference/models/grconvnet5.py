import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.models.grasp_model import GraspModel, ResidualBlock, CoordAtt, CARes, C3, Conv, C3_CA, Dense_Block, Conv_Mish, GSConv, GSBottleneck, VoVGSCSP,\
    Trans, VoVGSCSP_CA, VoVGSCSP_Res, GS_IRes


class GenerativeResnet(GraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=16, dropout=False, prob=0.0):
        super(GenerativeResnet, self).__init__()
        
        # 多任务可学习损失
        self.log_vars = nn.Parameter(torch.zeros(4))

        self.conv1 = Conv(input_channels, channel_size, 9, 1, 4 )
        self.conv2 = Conv(channel_size, channel_size*2, 3, 2, 1 )
        self.conv3 = Conv(channel_size*2, channel_size*4, 3, 2, 1 )

        # 1
        # self.gsb = GSBottleneck(channel_size*4, channel_size*4, 3, 2, 1)
        # self.gsres1 = VoVGSCSP_Res(channel_size*4, channel_size*4, n=3)
        # self.gsres2 = VoVGSCSP_Res(channel_size*4, channel_size*4, n=3)
        # self.gsres3 = VoVGSCSP_Res(channel_size*4, channel_size*4, n=3)

        # 2
        # self.gsconv1 = GSConv(channel_size*4, channel_size*4, 3, 2, 1)
        # self.vov = VoVGSCSP(channel_size*4, channel_size*4, n=2)
        # self.res1 = CARes(channel_size*4, channel_size*4)
        # self.res2 = CARes(channel_size*4, channel_size*4)
        # self.res3 = CARes(channel_size*4, channel_size*4)

        # 3
        # self.c3ca1 = C3_CA(channel_size*4, channel_size*4, 3, 2, 1)
        # self.vov = VoVGSCSP(channel_size*4, channel_size*4, n=3)
        # self.res1 = CARes(channel_size*4, channel_size*4)
        # self.res2 = CARes(channel_size*4, channel_size*4)
        # self.c3ca2 = C3_CA(channel_size*4, channel_size*4, 1, 1, 0)

        # 4
        # self.c3ca1 = C3_CA(channel_size*4, channel_size*4, 3, 2, 1)
        # self.c3ca2 = C3_CA(channel_size*4, channel_size*4, 3, 1, 1)
        # self.c3ca3 = C3_CA(channel_size*4, channel_size*4, 3, 1, 1)
        # self.c3ca4 = C3_CA(channel_size*4, channel_size*4, 3, 1, 1)
        # self.c3ca5 = C3_CA(channel_size*4, channel_size*4, 1, 1, 0)

        # 5
        self.conv4 = Conv(channel_size*4, channel_size*4, 3, 2, 1)
        self.res1 = CARes(channel_size*4, channel_size*4)
        self.res2 = CARes(channel_size*4, channel_size*4)
        self.res3 = CARes(channel_size*4, channel_size*4)
        self.res4 = CARes(channel_size*4, channel_size*4)
        self.res5 = CARes(channel_size*4, channel_size*4)

        # 6
        # self.dense1 = Dense_Block(channel_size*4, channel_size*4, 2)
        # self.dense2 = Dense_Block(channel_size*4, channel_size*4, 3, size_to_half=False)
        # self.dense3 = Dense_Block(channel_size*4, channel_size*4, 3, size_to_half=False)
        # self.dense4 = Dense_Block(channel_size*4, channel_size*4, 3, size_to_half=False)

        # 7
        # self.conv4 = Conv(channel_size*4,channel_size*4, 3, 2, 1)
        # self.res1 = ResidualBlock(channel_size*4, channel_size*4)
        # self.res2 = ResidualBlock(channel_size*4, channel_size*4)
        # self.res3 = ResidualBlock(channel_size*4, channel_size*4)
        # self.res4 = ResidualBlock(channel_size*4, channel_size*4)
        # self.res5 = ResidualBlock(channel_size*4, channel_size*4)

        # 8
        # self.gsb = GSBottleneck(channel_size*4, channel_size*4, 3, 2, 1)
        # self.vov_ca1 = VoVGSCSP_CA(channel_size*4, channel_size*4)
        # self.vov_ca2 = VoVGSCSP_CA(channel_size*4, channel_size*4, n=3)
        # self.vov_ca3 = VoVGSCSP_CA(channel_size*4, channel_size*4)
 
        # 9
        # self.gsb = GSBottleneck(channel_size*4, channel_size*4, 3, 2, 1)
        # self.vov_ca1 = VoVGSCSP_CA(channel_size*4, channel_size*4)
        # self.vov_ca2 = VoVGSCSP_CA(channel_size*4, channel_size*4)
        # self.vov_ca3 = VoVGSCSP_CA(channel_size*4, channel_size*4)
        # self.res1 = CARes(channel_size*4, channel_size*4)
        # self.res2 = CARes(channel_size*4, channel_size*4)

        # 10
        # self.gsconv1 = GSConv(channel_size*4, channel_size*8, 1, 1, 0)
        # self.gsires1 = GS_IRes(channel_size*8, channel_size*8, n=1)
        # self.gsires2 = GS_IRes(channel_size*8, channel_size*8, n=3)
        # self.gsires3 = GS_IRes(channel_size*8, channel_size*4, n=1)
        # self.gsconv2 = GSConv(channel_size*8, channel_size*4, 3, 2, 1)



        self.trans1 = Trans(channel_size*4, channel_size*2, 4, 2, 1, 0)
        self.trans2 = Trans(channel_size*2, channel_size, 4, 2, 1, 0)
        self.trans3 = Trans(channel_size, channel_size, 4, 2, 1, 1)
        self.trans4 = Trans(channel_size, channel_size, 9, 1, 4, 0, False)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        # 对模型中的卷积和转置卷积层进行xavier权值初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = self.conv2(x)
        x = self.conv3(x)

        # 1
        # x = self.gsb(x)
        # x = self.gsres1(x)
        # x = self.gsres2(x)
        # x = self.gsres3(x)

        # 2
        # x = self.gsconv1(x)
        # x = self.vov(x)
        # x = self.res1(x)
        # x = self.res2(x)
        # x = self.res3(x)

        # 3
        # x = self.c3ca1(x)
        # x = self.vov(x)
        # x = self.res1(x)
        # x = self.res2(x)
        # x = self.c3ca2(x)

        # 4
        # x = self.c3ca1(x)
        # x = self.c3ca2(x)
        # x = self.c3ca3(x)
        # x = self.c3ca4(x)
        # x = self.c3ca5(x)

        # 5
        x = self.conv4(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        # 6
        # x = self.dense1(x)
        # x = self.dense2(x)
        # x = self.dense3(x)
        # x = self.dense4(x)

        # 7
        # x = self.conv4(x)
        # x = self.res1(x)
        # x = self.res2(x)
        # x = self.res3(x)
        # x = self.res4(x)
        # x = self.res5(x)

        # 8
        # x = self.gsb(x)
        # x = self.vov_ca1(x)
        # x = self.vov_ca2(x)
        # x = self.vov_ca3(x)

        # 9
        # x = self.gsb(x)
        # x = self.vov_ca1(x)
        # x = self.vov_ca2(x)
        # x = self.vov_ca3(x)
        # x = self.res1(x)
        # x = self.res2(x)

        # 10 
        # x = self.gsconv1(x)
        # x = self.gsires1(x)
        # x = self.gsires2(x)
        # x = self.gsires3(x)
        # x = self.gsconv2(x)

        
        x = F.relu(self.trans1(x))
        x = F.relu(self.trans2(x))
        x = F.relu(self.trans3(x))
        x = self.trans4(x)

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output, self.log_vars
