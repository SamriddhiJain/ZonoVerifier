import torch
import torch.nn as nn
import numpy as np

DEVICE = 'cpu'

class Zonotope(object):
    def __init__(self, inputs, eps):

        """
        This initializes the Zonolayer class and is required once.

        Args:
            inputs (torch.tensor)  : shape (1,1,img_x(28),img_y(28))
            eps (float) :

        self.a_* is always of a type of torch.tensor to maintain consistency
        a_0_prev will be the actual value after each propagation
        a_1_prev will be size of a_0_prev*num_epsilons

        """
        self.input = inputs
        self.eps = eps
        self.actualShape = inputs.shape
        self.a_0_prev = inputs.clone().detach()
        self.pixels = self.actualShape[1]*self.actualShape[2]*self.actualShape[3]

        ## SAFE ##
        # for separate e across all 784 pixels
        self.a_1_prev = eps*torch.eye(784)
        self.a_1_prev_conv = eps*torch.eye(784)
        self.a_1_prev_conv = self.a_1_prev_conv.reshape(784, 1, 28, 28)

        self.checkInputBounds()
        self.a_0_prev = self.normalize(self.a_0_prev)
        self.a_1_prev = self.a_1_prev/0.3081
        self.a_1_prev_conv = self.a_1_prev_conv/0.3081

    def normalize(self, inputs):
        mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1)).to(DEVICE)
        sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1)).to(DEVICE)

        return (inputs - mean) / sigma

    def checkInputBounds(self):
        self.a_0_prev = self.a_0_prev.view(-1, self.a_0_prev.size(0))

        lower, upper = self.bounds()
        for i in range(self.a_0_prev.shape[0]):
            if lower[i] < 0:
                self.a_0_prev[i] = upper[i]/2
                self.a_1_prev[i, i] = upper[i]/2
            elif upper[i] > 1:
                self.a_0_prev[i] = (1 + lower[i])/2
                self.a_1_prev[i, i] = (1 - lower[i])/2

        self.a_0_prev = self.a_0_prev.reshape(self.actualShape)
        self.a_1_prev_conv =  self.a_1_prev.reshape(-1, self.actualShape[1], self.actualShape[2], self.actualShape[3])

    def bounds(self):
        abs_a_1 = torch.abs(self.a_1_prev).sum(dim=1)
        abs_a_1 = abs_a_1.reshape(abs_a_1.shape[0],1)

        lower = torch.cat([self.a_0_prev,-1*abs_a_1],1).sum(dim=1).detach().numpy()
        upper = torch.cat([self.a_0_prev,abs_a_1],1).sum(dim=1).detach().numpy()

        return lower, upper

class ZonoLearn(object):
    def __init__(self, zonotope):
        self.a_0_prev = zonotope.a_0_prev.clone().detach()
        self.a_1_prev = zonotope.a_1_prev.clone().detach()
        self.a_1_prev_conv = zonotope.a_1_prev_conv.clone().detach()
        self.actualShape = self.a_0_prev.shape
        self.isFCNext = False

    def flatten(self, layer):
        # update size
        self.a_0_prev = self.a_0_prev.view(-1, self.a_0_prev.size(0))
        self.actualShape = self.a_0_prev.shape
        nEpsilon = self.a_1_prev_conv.shape[0]
        self.a_1_prev = torch.transpose(self.a_1_prev_conv.view(nEpsilon, -1), 0, 1)
        self.isFCNext = True

    def conv_propogate(self, layer):
        nEpsilon = self.a_1_prev_conv.shape[1]
        convInput2 = self.a_1_prev_conv

        final = torch.cat((self.a_0_prev, convInput2), 0)

        output = layer(final)
        self.a_0_prev = output[0].unsqueeze(0)
        self.actualShape = self.a_0_prev.shape

        bias = layer.bias
        bias = bias.view(-1, 1, 1)
        biasExpand = bias.repeat(1, 1, self.actualShape[2], self.actualShape[3])

        term = output[1:]
        term = term - biasExpand

        self.a_1_prev_conv = term

    def linear_propogate(self, layer):
        # self.a_0_prev = torch.tensor([[4.0], [3.0]])
        # self.a_1_prev = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
        bias = layer.bias
        bias = bias.view(-1, 1)
        a_0_next = torch.matmul(layer.weight,self.a_0_prev) + bias
        a_1_next = torch.matmul(layer.weight,self.a_1_prev)

        self.a_0_prev = a_0_next
        self.a_1_prev = a_1_next

        self.actualShape = self.a_0_prev.shape

    def relu_propogate(self, para_lambda=None):
        if self.isFCNext:
            return self.relu_propogate_0_lambda(para_lambda)
        else:
            return self.relu_propogate_conv_0_lambda(para_lambda)

    def relu_propogate_conv_0_lambda(self, para_lambda=None):
        nEpsilon = self.a_1_prev_conv.shape[0]

        self.a_0_prev = torch.transpose(self.a_0_prev.view(1, -1), 0, 1)
        self.a_1_prev = torch.transpose(self.a_1_prev_conv.view(nEpsilon, -1), 0, 1)

        l, ind_relu, size = self.relu_propogate_0_lambda(para_lambda)

        t1 = torch.transpose(self.a_0_prev, 0, 1)
        t2 = torch.transpose(self.a_1_prev, 0, 1)
        self.a_1_prev_conv = t2.reshape(-1, self.actualShape[1], self.actualShape[2], self.actualShape[3])
        self.a_0_prev = t1.reshape(-1, self.actualShape[1], self.actualShape[2], self.actualShape[3])

        return l, ind_relu, size

    def relu_propogate_0_lambda(self, para_lambda=None):
        # Debugging
        # self.a_0_prev = torch.tensor([[1.0], [2.0], [3.0]])
        # self.a_1_prev = torch.tensor([[-2.0, 1], [-1, 2], [-1, 1]])
        # self.a_0_prev = torch.tensor([[10.0], [-1.0]])
        # self.a_1_prev = torch.tensor([[4.0, 5.0], [-1.0, 1.0]])
        lower, upper = self.bounds2()
        # reset_optim = False

        ind_0 = np.where(upper<=0)[0]
        ind_x = np.where(lower>=0)[0]
        ind_relu = np.where(np.logical_and(lower<0, 0<upper))[0]
        # print(len(ind_relu))
        a_0_next = torch.zeros(self.a_0_prev.shape)
        a_1_next = torch.zeros(self.a_1_prev.shape[0], self.a_1_prev.shape[1]+len(ind_relu))

        # a_0_next[ind_0,:] = 0
        a_0_next[ind_x,:] = self.a_0_prev[ind_x,:]
        self.a_1_prev[ind_0,:] = 0
        self.a_1_prev[ind_x,:] = self.a_1_prev[ind_x,:]
        l = torch.Tensor()

        if len(ind_relu) != 0:
            l_opt = torch.div(upper[ind_relu], upper[ind_relu]-lower[ind_relu]).reshape(-1)
            l = l_opt

            if para_lambda is not None:
                l = para_lambda[ind_relu]

            l = torch.clamp(l, 0, 1)
            l = l.reshape(-1)
            mask = (l <= l_opt)
            newError = torch.zeros(1, len(ind_relu))
            errorUpper = -torch.mul(l, lower[ind_relu])/2
            errorLower = torch.mul(1-l, upper[ind_relu])/2

            a_1_error = torch.zeros(self.a_1_prev.shape[0],len(ind_relu))
            newError = torch.mul(mask, errorLower) + torch.mul(~mask, errorUpper)

            a_1_error[ind_relu, :] = torch.mul(newError, torch.eye(len(ind_relu)))

            l = l.reshape(-1, 1)
            newError = newError.reshape(-1, 1)
            # print(torch.mul(l, self.a_0_prev[ind_relu,:]).shape, newError.shape)
            a_0_next[ind_relu, :] = torch.mul(l, self.a_0_prev[ind_relu,:]) + newError

            # i = 0
            # for each_index in ind_relu:
            #     a_0_next[each_index,:] = torch.mul(l[i],self.a_0_prev[each_index,:]) + newError[i]
            #     # a_1_error[each_index, i] = newError[i]
            #     i = i +1

            self.a_1_prev[ind_relu,:] = torch.mul(l,self.a_1_prev[ind_relu,:])

            a_1_next = torch.cat([self.a_1_prev, a_1_error],1)
            self.a_1_prev = a_1_next

        self.a_0_prev = a_0_next
        return l, ind_relu, self.a_0_prev.shape[0]

    def bounds(self):
        abs_a_1 = torch.abs(self.a_1_prev).sum(dim=1)
        abs_a_1 = abs_a_1.reshape(abs_a_1.shape[0],1)

        lower = torch.cat([self.a_0_prev,-1*abs_a_1],1).sum(dim=1).detach().numpy()
        upper = torch.cat([self.a_0_prev,abs_a_1],1).sum(dim=1).detach().numpy()

        return lower, upper

    def bounds2(self):
        abs_a_1 = torch.abs(self.a_1_prev.clone()).sum(dim=1)
        abs_a_1 = abs_a_1.reshape(abs_a_1.shape[0],1)

        lower = torch.cat([self.a_0_prev.clone(),-1*abs_a_1],1).sum(dim=1)
        upper = torch.cat([self.a_0_prev.clone(),abs_a_1],1).sum(dim=1)

        return lower, upper

    def verification(self, true_label):
        self.a_0_prev = self.a_0_prev - self.a_0_prev[true_label]
        self.a_1_prev = self.a_1_prev - self.a_1_prev[true_label, :]

        lower, upper = self.bounds()
        # print(lower, upper)
        if np.max(upper) <= 0:
            return True
        else:
            return False

    def getCostBounds(self, true_label):
        self.a_0_prev = self.a_0_prev - self.a_0_prev[true_label]
        self.a_1_prev = self.a_1_prev - self.a_1_prev[true_label, :]

        return self.bounds2()

    def verifyTemp(self, true_label):
        a0 = self.a_0_prev.clone().detach()
        a1 = self.a_1_prev.clone().detach()
        t0 = a0 - a0[true_label]
        t1 = a1 - a1[true_label, :]

        abs_a_1 = torch.abs(t1).sum(dim=1)
        abs_a_1 = abs_a_1.reshape(abs_a_1.shape[0],1)

        lower = torch.cat([t0,-1*abs_a_1],1).sum(dim=1)
        upper = torch.cat([t0,abs_a_1],1).sum(dim=1)
        # print(lower, upper)
        return (torch.max(upper) <= 0)
