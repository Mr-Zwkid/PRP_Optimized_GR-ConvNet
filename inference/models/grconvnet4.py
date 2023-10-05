import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.models.grasp_model import GraspModel, ResidualBlock, CoordAtt, CARes, C3, Conv, C3_CA, Dense_Block


class GenerativeResnet(GraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=16, dropout=False, prob=0.0):
        super(GenerativeResnet, self).__init__()
        
        # 多任务可学习损失
        self.log_vars = nn.Parameter(torch.zeros(4))

        self.conv1 = Conv(input_channels, channel_size, 9, 1, 4)
        self.conv2 = Conv(channel_size, channel_size * 2, 4, 2, 1)

        # 1
        # self.dense1 = Dense_Block(channel_size*2, channel_size*2, 3)
        # self.dense2 = Dense_Block(channel_size*2, channel_size*4, 3, size_to_half=False)
        # self.dense3 = Dense_Block(channel_size*4, channel_size*4, 3 )

        # 2
        self.dense1 = Dense_Block(channel_size*2, channel_size*2, 3)
        self.dense2 = Dense_Block(channel_size*2, channel_size*4, 2, size_to_half=False)
        self.dense3 = Dense_Block(channel_size*4, channel_size*4, 3 )
        self.dense4 = Dense_Block(channel_size*4, channel_size*4, 2, size_to_half=False)

        

        
        self.conv3 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1,
                                        )
        self.bn3 = nn.BatchNorm2d(channel_size * 2)
                
        self.conv4 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=1,
                                        )
        self.bn4 = nn.BatchNorm2d(channel_size)

        self.conv5 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=4, stride=2, padding=1, output_padding=1
                                        )
        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

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
        
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x) # 2
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

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
