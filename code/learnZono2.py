import argparse
import torch
from networks import FullyConnected, Conv
import torch.nn as nn
import numpy as np
import copy

from learnZono import ZonoLearn, Zonotope
DEVICE = "cpu"

# torch.manual_seed(0)
def getInputZonotope(inputs, eps):
    input = copy.deepcopy(inputs)
    eps1 = copy.deepcopy(eps)

    return Zonotope(input, eps1)

def model_forward(net, zonotope, true_label, relu_lambda=None, indexList=None):
    layer_zono = ZonoLearn(zonotope)
    index = []
    sizes = []
    lambdaGrad = []
    indVal = 0 # current starting index for lambda
    indCount = 0 # current relu layer

    for i, layer in enumerate(net.layers):
        if isinstance(layer,nn.Flatten):
            layer_zono.flatten(layer)
        if isinstance(layer,nn.Conv2d):
            layer_zono.conv_propogate(layer)
        if isinstance(layer,nn.Linear):
            layer_zono.linear_propogate(layer)
        elif isinstance(layer,nn.ReLU):
            if relu_lambda is not None:
                nextIndex = indVal + indexList[indCount]
                lambdaValues, indx_relu, size = layer_zono.relu_propogate(relu_lambda[indVal:nextIndex])
                indVal = nextIndex
                indCount += 1
            else:
                lambdaValues, indx_relu, size = layer_zono.relu_propogate()
            lambdaGrad = np.append(lambdaGrad, lambdaValues.detach().numpy())
            index = index + [indx_relu]
            sizes = sizes + [size]

    return torch.Tensor(lambdaGrad), index, layer_zono, sizes

def getPlaceholderLambda(net, oldLambda, index, sizes):
    lambdaNew = torch.zeros(np.sum(sizes))
    # lambdaNew = torch.rand(np.sum(sizes))

    reluOffset = 0
    oldIndexOffset = 0
    for i, indValues in enumerate(index):
        lambdaNew[reluOffset + indValues] = oldLambda[oldIndexOffset:oldIndexOffset + len(indValues)]
        reluOffset = reluOffset + sizes[i]
        oldIndexOffset = oldIndexOffset + len(indValues)

    return lambdaNew

def getOptimizerSchedular(lambdaNew, netName):
    learningRate = 0.04
    patience = 5
    factor = 0.5
    if netName == 'conv3':
        learningRate = 0.025
        patience = 2
        factor = 0.25
    elif netName == 'conv5':
        patience = 3

    optimizer = torch.optim.Adam([lambdaNew], lr=learningRate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                    factor=factor, min_lr=0.000001, threshold=0.001, patience=patience, verbose=False)
    return optimizer, scheduler

def analyze_zono(net, netName, inputs, eps, true_label, useRandom=False):
    zonotopeInput = getInputZonotope(inputs, eps)
    zn = copy.deepcopy(zonotopeInput)
    lambdaOpt, indexI, zono, sizes = model_forward(net, zn, true_label)

    if zono.verification(true_label):
        return True
    else:
        # call function to update lambda here
        lambdaGrad2 = getPlaceholderLambda(net, lambdaOpt.clone().detach(), indexI, sizes)
        lambdaGrad2 = lambdaGrad2.reshape(-1)

        for param in net.parameters():
            param.requires_grad = False
        # torch.autograd.set_detect_anomaly(True)
        iter = 10 if useRandom else 1
        for i in range(iter):
            if useRandom:
                lambdaGrad = lambdaGrad2.clone().detach() + torch.rand(lambdaGrad2.shape)
            else:
                lambdaGrad = lambdaGrad2.clone().detach()

            lambdaGrad = torch.clamp(lambdaGrad, 0, 1)
            lambdaGrad = lambdaGrad.reshape(-1)

            lambdaNew = lambdaGrad.clone()
            lambdaNew = lambdaNew.to(DEVICE)
            lambdaNew = lambdaNew.type(torch.float)
            lambdaNew.requires_grad_()

            optimizer, scheduler = getOptimizerSchedular(lambdaNew, netName)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
            # optimizer = torch.optim.Adamax([lambdaNew], lr=0.05)
            # optimizer = torch.optim.SGD([lambdaNew], lr=0.05, momentum=0.9)
            for j in range(500):
                optimizer.zero_grad()
                net.zero_grad()

                zn = copy.deepcopy(zonotopeInput)
                _, _, zono2, _ = model_forward(net, zn, true_label, lambdaNew, sizes)

                # if(zono2.verifyTemp(true_label)):
                #     return True

                lower, upper = zono2.getCostBounds(true_label)

                # s = nn.Sigmoid()
                # loss = torch.max(torch.zeros(upper.shape[0]), upper)
                # loss = torch.sum(loss)

                upper[true_label] = -10000
                loss = torch.max(upper)

                if(loss <= 0.0):
                    return True

                loss.backward()
                optimizer.step()
                scheduler.step(loss.item())
                # print(loss)

    return False
