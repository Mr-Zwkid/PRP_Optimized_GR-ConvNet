import torch
import torch.nn as nn
import torch.nn.functional as F


class GraspModel(nn.Module):
    """
    An abstract model for grasp network in a common format.
    """

    def __init__(self):
        super(GraspModel, self).__init__()

    def forward(self, x_in):
        raise NotImplementedError()

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc

        # 加入log_vars
        # pos_pred, cos_pred, sin_pred, width_pred = self(xc)
        pos_pred, cos_pred, sin_pred, width_pred, log_vars = self(xc)

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        # 1/sigma方 = exp(-log(sigma方))
        precision1, precision2, precision3, precision4 = \
            torch.exp(-log_vars[0]), torch.exp(-log_vars[1]), \
            torch.exp(-log_vars[2]), torch.exp(-log_vars[3])

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,  
            # ！！可适当调节三种loss的权重 最后有时间可以尝试
            # 'loss': p_loss / (2 * (sigma1 ** 2)) + cos_loss / (2 * (sigma2 ** 2) +sin_loss / (2 * (sigma3 ** 2) + width_loss / (2 * (sigma4 ** 2)  + math.log(sigma1[,base]) + math.log(sigma2[,base]) + math.log(sigma3[,base]) + math.log(sigma4[,base]),
            # 'loss': p_loss + 0.5 * (cos_loss + sin_loss) + width_loss,

            'lloss': 0.5 * (precision1 * p_loss + log_vars[0] + 
                            precision2 * cos_loss + log_vars[1] + 
                            precision3 * sin_loss + log_vars[2] + 
                            precision4 * width_loss + log_vars[3]),
            
            'weights': {
                'center_point': 0.5 * precision1,
                'cos': 0.5 * precision2,
                'sin': 0.5 * precision3,
                'width': 0.5 * precision4
            },

            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

    def predict(self, xc):
        pos_pred, cos_pred, sin_pred, width_pred, _ = self(xc)
        return {
            'pos': pos_pred,
            'cos': cos_pred,
            'sin': sin_pred,
            'width': width_pred
        }


class Conv(nn.Module):
    # ConvBNSiLU Module
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, g)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, k=1, s=1, p=0, shortcut=True, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, k, s, p)
        self.add = shortcut and c1 == c2 

    def forward(self, x_in):
        x = self.cv2(self.cv1(x_in))
        return x + x_in if self.add and x.shape == x_in.shape else x

class Dense_Layer(nn.Module):
    def __init__(self, c1, growth, e = 0.5): 
        super(Dense_Layer, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv1 = nn.Conv2d(c1, int(growth * e), kernel_size = 1, stride= 1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(growth * e))
        self.conv2 = nn.Conv2d(int(growth * e), growth, 3, 1, 1, bias=False)

    def forward(self, x_in):
        x = self.conv1(self.act(self.bn1(x_in)))
        x = self.conv2(self.act(self.bn2(x)))
        return torch.cat((x_in, x), 1)
    
class _Dense_Block(nn.Module):
    def __init__(self, c1, c2, n, e = 0.5):
        super(_Dense_Block, self).__init__()
        self.n = n
        self.layers = nn.Sequential(*(Dense_Layer(c1 + c2 * i, c2, e) for i in range(n)))
    def forward(self, x):
        x = self.layers(x)
        return x

class Transition(nn.Module): #size -> size/2
    def __init__(self, c1, c2, size_to_half = True):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(c1)
        self.conv = nn.Conv2d(c1, c2, 1, 1)
        self.act = nn.ReLU(True)
        self.pool = nn.AvgPool2d(2,2)
        self.flag = size_to_half
       
    def forward(self, x):
        x = self.conv(self.act(self.bn(x)))
        return self.pool(x) if self.flag else x

class Dense_Block(nn.Module): #size -> size/2
    def __init__(self, ch_in, ch_out, n, e = 0.5, growth = 32, size_to_half = True):
        super(Dense_Block, self).__init__()
        self.dense = _Dense_Block(n = n, c1 = ch_in, c2 = growth)
        self.tran = Transition(ch_in + growth * n, ch_out, size_to_half)

    def forward(self, x):
        return self.tran(self.dense(x))


class C3(nn.Module): 
    # ch1-> ch2; w -> w/2 ; h -> h/2 
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, k=1, s=1, p=0, n=1, shortcut=True, e=0.5):
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, k, s, p)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, k, s, p, shortcut, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
    

class C3_CA(nn.Module): 
    # ch1-> ch2; w -> w/2 ; h -> h/2  if k = 4, s = 2, p = 1
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, k=1, s=1, p=0, n=1, shortcut=True, e=0.5):
        super(C3_CA, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, k, s, p)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, k, s, p, shortcut, e=1.0) for _ in range(n)))
        self.ca = CoordAtt(c2, c2)

    def forward(self, x):
        x1 = self.cv1(x)
        x1 = self.m(x1)
        x2 = self.cv2(x)
        x3 = torch.concat((x1,x2),1)
        x3 = self.cv3(x3)
        x3 = self.ca(x3)
        out = F.relu(x3)

        return out  



class ResidualBlock(nn.Module):
    """
    A residual block with dropout option
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in # 残差块的实现


# 加入 SE bias
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# CA Attention
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class CARes(nn.Module):

    def __init__(self, c1, c2, reduction=32):
        super(CARes, self).__init__()
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=3, stride = 1, padding=1)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, kernel_size=3, stride = 1, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)

        self.relu = nn.ReLU(inplace=True)
        self.ca = CoordAtt(c1, c2, reduction=reduction)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out)

        out += residual
        out = self.relu(out)

        return out


# ---------------------------GSConv Begin---------------------------
class Conv_Mish(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True): 
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Mish() if act else nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
class GSConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act = True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv_Mish(c1, c_, k, s, p, g, act)
        self.cv2 = Conv_Mish(c_, c_, 3, 1, 1, c_, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)

        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)


class GSBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1, 0),
            GSConv(c_, c2, k, s, p))

    def forward(self, x_in):
        x = self.conv_lighting(x_in)
        return x + x_in if x.shape == x_in.shape else x
    

# class GS_IRes(nn.Module):
#     def __init__(self, c1, c2, n = 1, e = 0.5, RES = True):
#         super().__init__() 
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=0.5) for _ in range(n)))
#         self.cv3 = Conv(2 * c_, c2, 1)  
#         self.cv4 = Conv(c2, c1, 1, 1, 0)
#         self.res = RES

#     def forward(self, x):
#         y1 = self.gsb(self.cv1(x))
#         y2 = self.cv2(x)
#         y = self.cv4(torch.concat((y1, y2), 1))
#         out = y + x if x.shape == y.shape and self.res else self.cv4(y) + x if self.res else y
#         return out

class GS_IRes(nn.Module):
    def __init__(self, c1, c2, n = 1, e = 0.5, RES = True):
        super().__init__() 
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=0.5) for _ in range(n)))

        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1, 1)   
        self.cv4 = Conv(c_, c_, 3, 1, 1)   

        self.cv5 = Conv(c1, c_, 1, 1)  

        self.cv6 = Conv(3 * c_, c1, 1, 1, 0)  
        self.res = RES
        self.act = nn.ReLU(inplace = True)

    def forward(self, x):
        y1 = self.gsb(self.cv1(x))
        y2 = self.cv4(self.cv3(self.cv2(x)))
        y3 = self.cv5(x)

        y = self.cv6(torch.concat((y1, y2, y3), 1))

        out = y + x if self.res else y
        return out 
        # return self.act(out)

class CA_GS_IRes(nn.Module):
    def __init__(self, c1, c2, n = 1, e = 0.5, RES = True):
        super().__init__() 
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=0.5) for _ in range(n)))

        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1, 1)   
        self.cv4 = Conv(c_, c_, 3, 1, 1)   

        self.cv5 = Conv(c1, c_, 1, 1)  

        self.cv6 = Conv(3 * c_, c1, 1, 1, 0)
        self.ca = CoordAtt(c1, c1)  
        self.res = RES
        self.act = nn.ReLU(inplace = True)

    def forward(self, x):
        y1 = self.gsb(self.cv1(x))
        y2 = self.cv4(self.cv3(self.cv2(x)))
        y3 = self.cv5(x)

        y = self.cv6(torch.concat((y1, y2, y3), 1))
        y = self.ca(y)
        out = y + x if self.res else y
        return self.act(out)


class VoVGSCSP(nn.Module):
    # VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
        self.cv3 = Conv(2 * c_, c2, 1)  

    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))
    
class VoVGSCSP_Res(nn.Module):
    # VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
        self.cv3 = Conv(c_, c2, 1)  

    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(x1 + y)
    
class VoVGSCSP_CA(nn.Module):
    # VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
        self.cv3 = Conv(2 * c_, c2, 1)  
        self.ca1 = CoordAtt(c_, c_)
        self.ca2 = CoordAtt(c2, c2)

    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        x1 = self.ca1(x1)
        return self.cv3(torch.cat((y, x1), dim=1))

# ---------------------------GSConv End---------------------------

class Trans(nn.Module):
    def __init__(self, c1, c2, k = 4, s = 2, p1 = 1, p2 = 0, bn = True):
        super(Trans, self).__init__()
        self.conv = nn.ConvTranspose2d(c1, c2, k, s, p1, p2)
        self.bn = nn.BatchNorm2d(c2)
        self.flag = bn
    def forward(self, x):
        return self.bn(self.conv(x)) if self.flag else self.conv(x)
    
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))
    
class MultiModal(nn.Module):
    def __init__(self, c1, c2, k = 7, s = 1, p = 3):
        super(MultiModal, self).__init__()
        self.conv = Conv_Mish(c1, c1, k, s, p, c1)
        self.gsconv = GSConv(c1, c2, k, s, p)
        self.ca = CoordAtt(c2, c2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gsconv(x)
        x = self.ca(x)
        x = self.act(x)
        return x
        


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 将输入张量的形状从 [batch_size, in_channels, height, width] 转换为 [batch_size, height*width, in_channels]
        batch_size, _, height, width = x.size()
        x = x.view(batch_size, self.in_channels, height * width).permute(0, 2, 1)

        # 计算注意力权重
        q = self.query(x)
        k = self.key(x)
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = self.softmax(attn_weights)

        # 使用注意力权重对值进行加权求和
        v = self.value(x)
        weighted_values = torch.matmul(attn_weights, v)

        # 将加权求和后的结果重新转换为 [batch_size, in_channels, height, width] 的形状
        weighted_values = weighted_values.permute(0, 2, 1).view(batch_size, self.in_channels, height, width)

        # 返回加权求和后的结果
        return weighted_values

