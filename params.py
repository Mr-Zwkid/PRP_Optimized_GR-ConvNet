# import torch
# from torchvision.models import resnet18
# import sys
# from pthflops import count_ops
# from torchsummary import summary
#
# # Create a network and a corresponding input
# device = 'cuda:0'
# model = resnet18().to(device)
# inp = torch.rand(1,3,224,224).to(device)
#
# summary(model, input_size=(4, 224, 224), batch_size=-1)
# sys.stdout = sys.__stdout__
#
# # Count the number of FLOPs

indices = list(range(10))
print(indices[8:])
