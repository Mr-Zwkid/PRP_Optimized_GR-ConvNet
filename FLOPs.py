import torch.nn.functional as F
import torch
from torch import nn
# from fvcore.nn.flop_count import flop_count
# from fvcore.nn import FlopCountAnalysis, parameter_count_table
# from inference.models.grconvnet2 import GenerativeResnet



# device = 'cuda'
# tensor = torch.ones((1, 4, 224, 224), dtype=torch.float32).to(device)
# net = GenerativeResnet().to(device)

# # flops, _ = flop_count(net, tensor)
# # print(flops)
# # print("Total FLOPs: {:.2f} G".format(int(flops) / 10**9))


# # 分析FLOPs
# flops = FlopCountAnalysis(net, tensor)
# print("FLOPs: ", flops.total())

# # # 分析parameters
# # print(parameter_count_table(net))
