import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch import nn
# cost = np.array([[1, 1], [2, 4]])
# from scipy.optimize import linear_sum_assignment
# row_ind, col_ind = linear_sum_assignment(cost)
# print(col_ind)
#
# x = torch.Tensor([[2,2,2,2],[1,1,1,1]])
# print(j for j in x)
#
# test = np.zeros(13)
# print(test.tolist())

# aa = torch.Tensor([[0.6889, 0.2743, 0.7347, 0.0000, 0.1192],
#         [0.3569, 0.3785, 0.4181, 0.0000, 0.4637],
#         [0.8792, 0.4201, 0.9472, 0.0000, 0.3423],
#         [0.8792, 0.4201, 0.9472, 0.0000, 0.3423],
#         [0.8792, 0.4201, 0.9472, 0.0000, 0.3423],
#         [0.8792, 0.4201, 0.9472, 0.0000, 0.3423]])
#
# bb = torch.Tensor([[0.6889, 0.2743, 0.7347, 0.0000, 0.1192],
#         [0.3569, 0.3785, 0.4181, 0.0000, 0.4637],
#         [0.8792, 0.4201, 0.9472, 0.0000, 0.3423],
#         [0.8792, 0.4201, 0.9472, 0.0000, 0.3423],
#         [0.8792, 0.4201, 0.9472, 0.0000, 0.3423],
#         [0.8792, 0.4201, 0.9472, 0.0000, 0.3423]])
#
# print(torch.cat((aa,bb), dim=0))

# dist = Categorical(logits=torch.Tensor([[1,2,3,4],[1,2,3,4]]))
# for i in range(20):
#         print(dist.sample())
x_input=torch.randn(2,4,16)
print('x_input:\n',x_input)
softmax_func=nn.Softmax(dim=1)
soft_output=softmax_func(x_input)
print('soft_output:\n',soft_output)
y_target=torch.tensor([[1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,2], [1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,2]])
print(y_target.shape)

crossentropyloss=nn.CrossEntropyLoss()
crossentropyloss_output=crossentropyloss(x_input,y_target)
print('crossentropyloss_output:\n',crossentropyloss_output)


