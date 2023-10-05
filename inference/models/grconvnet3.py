import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.models.grasp_model import GraspModel, ResidualBlock, SELayer, h_sigmoid, h_swish, CoordAtt, CARes, C3, Conv


# # 多任务损失封装
# class GenerativeResnet(GraspModel):
#
#     def __init__(self, task_num, model):
#         super(GenerativeResnet, self).__init__()
#         self.model = model
#         self.task_num = task_num
#         self.log_vars = nn.Parameter(torch.zeros((task_num)))
#
#     def forward(self, input):
#
#         pos_output, cos_output, sin_output, width_output = self.model(input)
#
#         return pos_output, cos_output, sin_output, width_output


class GenerativeResnet(GraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=16, dropout=False, prob=0.0):
        super(GenerativeResnet, self).__init__()

        # 多任务可学习损失
        self.log_vars = nn.Parameter(torch.zeros(4))

        # 测试鲁棒性

        # self.log_vars = nn.Parameter(torch.tensor([-4.889897899, -4.854527144, -4.33283639, -3.77505715]).float())
        # self.log_vars = nn.Parameter(torch.tensor([-4.094344562, -3.77505715, -4.854527144, -4.33283639]).float())
        # self.log_vars = nn.Parameter(torch.tensor([-5.298317367, 0, 0, 0]).float()) # 第一次
        # self.log_vars = nn.Parameter(torch.tensor([0, -5.298317367, 0, 0]).float()) # 2
        # self.log_vars = nn.Parameter(torch.tensor([-5.298317367, -5.298317367, 0, 0]).float())  # 3
        # self.log_vars = nn.Parameter(torch.tensor([-5.298317367, -5.298317367, 0, -5.298317367]).float()) # 4
        # self.log_vars = nn.Parameter(torch.tensor([-5.298317367, 0, 0, -5.298317367]).float())  # 5
        # self.log_vars = nn.Parameter(torch.tensor([0, -5.298317367, 0, -5.298317367]).float())  # 6
        # self.log_vars = nn.Parameter(torch.tensor([0, 0, 0, -5.298317367]).float())  # 7
        # self.log_vars = nn.Parameter(torch.tensor([-5.298317367, -5.298317367, -5.298317367, 0]).float())  # 8
        # self.log_vars = nn.Parameter(torch.tensor([-5.298317367, 0, -5.298317367, 0]).float())  # 9
        # self.log_vars = nn.Parameter(torch.tensor([0, -5.298317367, -5.298317367, 0]).float())  # 10
        # self.log_vars = nn.Parameter(torch.tensor([0, 0, -5.298317367, 0]).float())  # 11
        # self.log_vars = nn.Parameter(torch.tensor([-5.298317367, -5.298317367, -5.298317367, -5.298317367]).float())  # 12
        # self.log_vars = nn.Parameter(torch.tensor([-5.298317367, 0, -5.298317367, -5.298317367]).float())  # 13
        # self.log_vars = nn.Parameter(torch.tensor([0, -5.298317367, -5.298317367, -5.298317367]).float())  # 14
        # self.log_vars = nn.Parameter(torch.tensor([0, 0, -5.298317367, -5.298317367]).float())  # 15

        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size) 
        
        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)

        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)

        # self.C3_2 = C3(channel_size, channel_size*2, k=3, s=2, p=1)
        # self.C3_3 = C3(channel_size*2, channel_size*4, k=3, s=2, p=1)
        # self.C3_3_1 = C3(channel_size*4, channel_size*4, n=3)

        # # SE
        # self.res1 = SEBottleneck(channel_size * 4, channel_size * 4)
        # self.res2 = SEBottleneck(channel_size * 4, channel_size * 4)
        # self.res3 = SEBottleneck(channel_size * 4, channel_size * 4)
        # self.res4 = SEBottleneck(channel_size * 4, channel_size * 4)
        # self.res5 = SEBottleneck(channel_size * 4, channel_size * 4)

        # Res
        self.res1 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res2 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res3 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res4 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res5 = ResidualBlock(channel_size * 4, channel_size * 4)

        # CARes
        self.res1 = CARes(channel_size * 4, channel_size * 4)
        self.res2 = CARes(channel_size * 4, channel_size * 4)
        self.res3 = CARes(channel_size * 4, channel_size * 4)
        self.res4 = CARes(channel_size * 4, channel_size * 4)
        self.res5 = CARes(channel_size * 4, channel_size * 4)

        self.conv4 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size * 2)

        self.conv5 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2,
                                        output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        # 224 * 224

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        # 对模型中的卷积和转置卷积层进行xavier权值初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)
                # nn.init.kaiming_normal_(m.weight)

    def forward(self, x_in):
        # 1
        # x = F.silu(self.bn1(self.conv1(x_in)))     
        # x = self.C3_2(x)
        # x = self.C3_3(x)
        # x = self.C3_3_1(x)

        # x = self.res1(x)
        # x = self.res2(x)
        # x = self.res3(x)
        # x = self.res4(x)
        # x = self.res5(x)

        # x = F.silu(self.bn4(self.conv4(x)))
        # x = F.silu(self.bn5(self.conv5(x)))
        # x = self.conv6(x)

        # 2
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))        
        

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

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

        # return pos_output, cos_output, sin_output, width_output
        return pos_output, cos_output, sin_output, width_output, self.log_vars
