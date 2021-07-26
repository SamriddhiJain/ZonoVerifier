import torch
import torch.nn as nn
import numpy as np

DEVICE = 'cpu'

class ZonoLayer(object):
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
        ## DON'T USE ##
        # for shared e1 across all 784 pixels
        # self.a_1_prev = eps*torch.ones(self.a_0_prev.shape[0],1)
        # self.a_1_prev_conv = (eps)*torch.ones(28, 28)
        # self.a_1_prev_conv = self.a_1_prev_conv.reshape(-1, 1, self.actualShape[2], self.actualShape[3])

        ## SAFE ##
        # for separate e across all 784 pixels
        self.a_1_prev = eps*torch.eye(784)
        self.a_1_prev_conv = eps*torch.eye(784)
        self.a_1_prev_conv = self.a_1_prev_conv.reshape(784, 1, 28, 28)

        ## DON'T USE ##
        # 784 epsilon shared among all pixels
        # self.a_1_prev = (eps/self.pixels)*torch.ones(self.pixels, self.pixels)
        # self.a_1_prev_conv = (eps/self.pixels)*torch.ones(self.pixels, self.pixels)
        # self.a_1_prev_conv = self.a_1_prev_conv.reshape(-1, 1, self.actualShape[2], self.actualShape[3])

        ## IMPROVED ##
        # two epsilons: one common among all 784, other separate
        # aTemp1 = (eps/2)*torch.ones(784,1)
        # aTemp2 = (eps/2)*torch.eye(784)
        # self.a_1_prev = torch.cat([aTemp1, aTemp2], 1)
        # convTemp1 = aTemp1.reshape(1, 1, 28, 28)
        # convTemp2 = aTemp2.reshape(784, 1, 28, 28)
        # self.a_1_prev_conv = torch.cat((convTemp1, convTemp2), 0)

        # self.num_error_terms = 1
        self.isFCNext = False
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
        # a_0_next = torch.matmul(layer.weight,self.a_0_prev) \
        #                     + layer.bias[:,np.newaxis]
        bias = layer.bias
        bias = bias.view(-1, 1)
        a_0_next = torch.matmul(layer.weight,self.a_0_prev) + bias

        a_1_next = torch.matmul(layer.weight,self.a_1_prev)

        self.a_0_prev = a_0_next
        self.a_1_prev = a_1_next

        self.actualShape = self.a_0_prev.shape

    def relu_propogate(self, para_lambda=None, prev_ind_relu=None):
        if self.isFCNext:
            return self.relu_propogate_0_lambda(para_lambda, prev_ind_relu)
        else:
            return self.relu_propogate_conv_0_lambda(para_lambda, prev_ind_relu)

    def relu_propogate_conv_0_lambda(self, para_lambda=-1.0, prev_ind_relu=None):
        nEpsilon = self.a_1_prev_conv.shape[0]

        self.a_0_prev = torch.transpose(self.a_0_prev.view(1, -1), 0, 1)
        self.a_1_prev = torch.transpose(self.a_1_prev_conv.view(nEpsilon, -1), 0, 1)

        l, indx, flag = self.relu_propogate_0_lambda(para_lambda, prev_ind_relu)

        t1 = torch.transpose(self.a_0_prev, 0, 1)
        t2 = torch.transpose(self.a_1_prev, 0, 1)
        self.a_1_prev_conv = t2.reshape(-1, self.actualShape[1], self.actualShape[2], self.actualShape[3])
        self.a_0_prev = t1.reshape(-1, self.actualShape[1], self.actualShape[2], self.actualShape[3])

        return l, indx, flag

    def relu_propogate_0_lambda(self, para_lambda=None, prev_ind_relu=None):
        # Debugging
        # self.a_0_prev = torch.tensor([[1.0], [2.0], [3.0]])
        # self.a_1_prev = torch.tensor([[-2.0, 1], [-1, 2], [-1, 1]])
        # self.a_0_prev = torch.tensor([[10.0], [-1.0]])
        # self.a_1_prev = torch.tensor([[4.0, 5.0], [-1.0, 1.0]])
        lower, upper = self.bounds()
        reset_optim = False

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
            # the new error term, size: N*num_of_new_e
            l_opt = torch.Tensor(np.divide(upper[ind_relu], upper[ind_relu]-lower[ind_relu])[:,np.newaxis])

            l = 0.3*l_opt
            if para_lambda is not None:
                if np.array_equal(ind_relu, prev_ind_relu):
                    # l = para_lambda
                    l = torch.clamp(para_lambda, 0, 1)
                else:
                    for i in range(len(ind_relu)):
                        idx = np.where(prev_ind_relu == ind_relu[i])
                        if len(idx) > 0 and len(idx[0]) > 0:
                            # print(para_lambda)
                            l[i] = para_lambda[idx[0][0]]

                    reset_optim = True

            newError = torch.tensor([-each_l*each_lower/2.0 if each_l>=each_l_opt else each_upper*(1-each_l)/2.0
                        for (each_l_opt, each_l, each_upper, each_lower) in
                        zip(l_opt,l, upper[ind_relu], lower[ind_relu])]).reshape(1,len(ind_relu))
            # print(newError)

            a_1_error = np.zeros((self.a_1_prev.shape[0],len(ind_relu)))

            #newError = -np.multiply(np.transpose(l),lower[ind_relu]/2.0)

            i = 0
            for each_index in ind_relu:
                a_0_next[each_index,:] = torch.mul(l[i],self.a_0_prev[each_index,:]) + newError[0, i]
                a_1_error[each_index, i] = newError[0, i]
                i = i +1

            self.a_1_prev[ind_relu,:] = torch.Tensor(torch.mul(l,self.a_1_prev[ind_relu,:]))

            a_1_next = torch.cat([self.a_1_prev,torch.Tensor(a_1_error)],1)
            self.a_1_prev = a_1_next

        self.a_0_prev = a_0_next
        # print(self.a_0_prev, self.a_1_prev)
        return l, ind_relu, reset_optim

    def bounds(self):
        abs_a_1 = torch.abs(self.a_1_prev).sum(dim=1)
        abs_a_1 = abs_a_1.reshape(abs_a_1.shape[0],1)

        lower = torch.cat([self.a_0_prev,-1*abs_a_1],1).sum(dim=1).detach().numpy()
        upper = torch.cat([self.a_0_prev,abs_a_1],1).sum(dim=1).detach().numpy()

        return lower, upper

    def bounds2(self):
        abs_a_1 = torch.abs(self.a_1_prev).sum(dim=1)
        abs_a_1 = abs_a_1.reshape(abs_a_1.shape[0],1)

        lower = torch.cat([self.a_0_prev,-1*abs_a_1],1).sum(dim=1)
        upper = torch.cat([self.a_0_prev,abs_a_1],1).sum(dim=1)

        return lower, upper

    def verification(self, true_label):
        self.a_0_prev = self.a_0_prev - self.a_0_prev[true_label]
        self.a_1_prev = self.a_1_prev - self.a_1_prev[true_label, :]

        lower, upper = self.bounds()
        print(lower, upper)
        if np.max(upper) <= 0:
            return True
        else:
            return False

    def getCostBounds(self, true_label):
        # self.a_0_prev = self.a_0_prev - self.a_0_prev[true_label]
        # self.a_1_prev = self.a_1_prev - self.a_1_prev[true_label, :]

        return self.bounds2()
