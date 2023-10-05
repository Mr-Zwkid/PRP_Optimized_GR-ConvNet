import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.models.grasp_model import GraspModel, ResidualBlock, CoordAtt, CARes, C3, Conv, C3_CA, Dense_Block, Conv_Mish, GSConv, GSBottleneck, VoVGSCSP,\
    Trans, VoVGSCSP_CA, VoVGSCSP_Res, GS_IRes,CA_GS_IRes, MultiModal


class GenerativeResnet(GraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=16, dropout=False, prob=0.0):
        super(GenerativeResnet, self).__init__()
        
        # 多任务可学习损失
        self.log_vars = nn.Parameter(torch.zeros(4))

        self.conv1 = Conv(input_channels, channel_size, 9, 1, 4 )
        self.multi_conv1 = MultiModal(input_channels, channel_size, 7, 1, 3)
        self.conv2 = Conv(channel_size, channel_size*2, 4, 2, 1 )
        self.conv3 = Conv(channel_size*2, channel_size*4, 4, 2, 1 )

        # 1
        # self.gsires1 = GS_IRes(channel_size*4, channel_size*4, 1)
        # self.gsires2 = GS_IRes(channel_size*4, channel_size*4, 1)
        # self.gsires3 = GS_IRes(channel_size*4, channel_size*4, 1)
        # self.gsires4 = GS_IRes(channel_size*4, channel_size*4, 1)
        # self.gsires5 = GS_IRes(channel_size*4, channel_size*4, 1)

        # 2
        # self.gsires1 = GS_IRes(channel_size*4, channel_size*4, 2)
        # self.gsires2 = GS_IRes(channel_size*4, channel_size*4, 2)
        # self.gsires3 = GS_IRes(channel_size*4, channel_size*4, 2)

        # 3
        self.ca_gsires1 = CA_GS_IRes(channel_size*4, channel_size*4, 2)
        self.ca_gsires2 = CA_GS_IRes(channel_size*4, channel_size*4, 2)
        self.ca_gsires3 = CA_GS_IRes(channel_size*4, channel_size*4, 2)



        self.trans1 = Trans(channel_size*4, channel_size*2, 4, 2, 1, 1)
        self.trans2 = Trans(channel_size*2, channel_size, 4, 2, 2, 1)
        self.trans3 = Trans(channel_size, channel_size, 9, 1, 4, 0, False)

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


        # 1
        # x = self.conv1(x_in)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.gsires1(x)
        # x = self.gsires2(x)
        # x = self.gsires3(x)
        # x = self.gsires4(x)
        # x = self.gsires5(x)

        # 2
        # x = self.conv1(x_in)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.gsires1(x)
        # x = self.gsires2(x)
        # x = self.gsires3(x)

        # 3
        # x = self.conv1(x_in)
        x = self.multi_conv1(x_in)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.ca_gsires1(x)
        x = self.ca_gsires2(x)
        x = self.ca_gsires3(x)

   
        
        x = F.relu(self.trans1(x))
        x = F.relu(self.trans2(x))
        x = self.trans3(x)

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
