import argparse
import torch
from networks import FullyConnected, Conv
import torch.nn as nn
import numpy as np
import copy

# from graphviz import Digraph
# # import torch
# # from torch.autograd import Variable
# # make_dot was moved to https://github.com/szagoruyko/pytorchviz
# from torchviz import make_dot

from zonotope import ZonoLayer
DEVICE = "cpu"

def model_forward(net, inputs, eps, true_label, relu_lambda=None, indexList=None):
    input = copy.deepcopy(inputs)
    eps1 = copy.deepcopy(eps)
    layer_zono = ZonoLayer(input, eps1)
    index = []
    lambdaGrad = []
    indVal = 0 # current starting index for lambda
    indCount = 0 # current relu layer
    resetFlag = False

    for i, layer in enumerate(net.layers):
        if isinstance(layer,nn.Flatten):
            layer_zono.flatten(layer)
        if isinstance(layer,nn.Conv2d):
            layer_zono.conv_propogate(layer)
        if isinstance(layer,nn.Linear):
            layer_zono.linear_propogate(layer)
        elif isinstance(layer,nn.ReLU):
            if relu_lambda is not None:
                nextIndex = indVal + len(indexList[indCount])
                lambdaValues, indx_relu, flag = layer_zono.relu_propogate(relu_lambda[indVal:nextIndex,:], indexList[indCount])
                # lambdaValues, indx_relu, flag = layer_zono.relu_propogate(relu_lambda)
                indVal = nextIndex
                indCount += 1
                resetFlag = resetFlag or flag
            else:
                lambdaValues, indx_relu, v1 = layer_zono.relu_propogate()
            lambdaGrad = np.append(lambdaGrad, lambdaValues.detach().numpy())
            index = index + [indx_relu]

    return torch.Tensor(lambdaGrad), index, layer_zono, resetFlag

def analyze_zono(net, inputs, eps, true_label, relu_lambda):
    lambdaGrad2, indexI, zono, flag = model_forward(net, inputs, eps, true_label)

    if zono.verification(true_label):
        return True
    else:
        lambdaGrad2 = lambdaGrad2.reshape(-1, 1)

        for param in net.parameters():
            param.requires_grad = False

        for i in range(10):
            index = copy.deepcopy(indexI)
            lambdaGrad = lambdaGrad2.clone().detach() + torch.rand(lambdaGrad2.shape)
            lambdaGrad = torch.clamp(lambdaGrad, 0, 1)
            lambdaGrad = lambdaGrad.reshape(-1, 1)

            lambdaNew = lambdaGrad.clone()
            lambdaNew = lambdaNew.to(DEVICE)
            lambdaNew = lambdaNew.type(torch.float)
            lambdaNew.requires_grad_()
            optimizer = torch.optim.Adam([lambdaNew], lr=0.05)
            # optimizer = torch.optim.SGD([lambdaNew], lr=0.005, momentum=0.9)
            for i in range(500):
                if flag:
                    print("cwecew")
                    flag = False
                    lambdaNew = lambdaGrad.clone()
                    lambdaNew = lambdaNew.to(DEVICE)
                    lambdaNew = lambdaNew.type(torch.float)
                    # lambdaNew = lambdaNew.reshape(-1, 1)
                    lambdaNew.requires_grad_()
                    optimizer= torch.optim.Adam([lambdaNew],lr=0.005, weight_decay=0.1)

                optimizer.zero_grad()
                net.zero_grad()

                lambdaNew2, index2, zono2, flag = model_forward(net, inputs, eps, true_label, lambdaNew, index)
                index = copy.deepcopy(index2)
                # make_dot(lambdaNew).view()
                lower, upper = zono2.getCostBounds(true_label)
                # if(torch.max(upper) <= 0.0):
                #     print("Solved")
                #     return True
                #
                # if flag:
                #     lambdaGrad = torch.reshape(lambdaNew2.detach(), (-1, 1))
                #     print("cwecew")
                #     continue

                loss = -lower[true_label]
                # s = nn.Sigmoid()
                # loss = y(torch.max(torch.zeros(upper.shape[0]), upper)) - 0.5
                # loss = torch.sum(torch.mul(loss, loss))
                # loss = torch.sum(y(-torch.max(torch.zeros(upper.shape[0]), upper)) - 0.5)
                # loss = y(-torch.max(torch.zeros(upper.shape[0]), upper) - 0.5)
                # loss = torch.sum(torch.mul(loss, loss))

                # mask = (upper != 0.0)
                # upper2 = upper[torch.nonzero(mask)]
                # upper2 = torch.reshape(upper2, (-1, 1))
                # s = nn.Sigmoid()
                # loss = torch.sum(s(upper2))

                # pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
                # print(pytorch_total_params)
                # y = nn.Softmax()
                # diff = y(upper)
                # diff = torch.reshape(diff, (1, -1))
                # target = torch.LongTensor([true_label]).to(DEVICE)
                # loss = nn.CrossEntropyLoss()(diff, target)
                # loss = y(loss) - 0.5
                # loss = torch.sum(upper > 0)
                # loss = torch.sum(y(-1*torch.min(torch.zeros(lower.shape[0]), lower))-0.5)
                # loss = y(torch.sum(torch.max(torch.zeros(upper.shape[0]), upper)))
                # print(lower)
                # loss = 1/torch.sum(y(torch.min(torch.zeros(lower.shape[0]), lower))-0.5)
                # optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(loss, lower)
                # lambdaNew.zero_grad()
                # lambdaNew = torch.clamp(lambdaNew, 0, 1)
                # lambdaNew = lambdaNew - 0.05*lambdaNew.grad
                # lambdaNew = torch.clamp(lambdaNew, 0, 1)
            print("end i")

    return False
