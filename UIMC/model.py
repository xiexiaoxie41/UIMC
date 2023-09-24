import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributions as td
import numpy
import matplotlib.pyplot as plt


# loss function
def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    # beta = torch.ones((1, c))
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


# Using the Expected Cross Entropy
def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    return (A + B)



class UGC(nn.Module):
    def __init__(self, classes, views, classifier_dims, annealing_epochs=1):
        super(UGC, self).__init__()
        self.views = views
        self.classes = classes
        self.annealing_epochs = annealing_epochs
        self.Net = nn.ModuleList([Net(classifier_dims[i], self.classes) for i in range(self.views)])

    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        def DS_Combin_two(alpha1, alpha2):
            """
            Dempster’s  combination  rule  for  two  independent  sets  of  masses
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = self.classes/S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, self.classes, 1), b[1].view(-1, 1, self.classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))
            # test = torch.sum(b_a, dim = 1, keepdim = True) + u_a #Verify programming errors

            # calculate new S
            S_a = self.classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha)-1):
            if v==0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
        return alpha_a

    def classify(self, input , y, global_step, sn):
        # pretrain
        evidence, sigma = self.collect(input)
        loss_class = 0
        alpha = dict()
        for v_num in range(self.views):
            sn_v = torch.unsqueeze(sn[:, v_num], 1)
            alpha[v_num] = evidence[v_num] + 1
            loss_class += ce_loss(y, alpha[v_num], self.classes, global_step, self.annealing_epochs) * sn_v
        loss_class = torch.mean(loss_class)
        return loss_class


    def forward(self, input, y, global_step, batch_idx, sn, if_test=0):
        if(if_test):
            evidence = self.collect(input)
            alpha = dict()
            for v_num in range(self.views):
                alpha[v_num] = evidence[v_num] + 1
            alpha_a = self.DS_Combin(alpha)
            evidence_a = alpha_a - 1
            return evidence, evidence_a
        # 分类器损失
        evidence = self.collect(input)
        loss = 0
        alpha = dict()
        for v_num in range(self.views):
            # sn_v = torch.unsqueeze(sn[:, v_num], 1)
            alpha[v_num] = evidence[v_num] + 1
            loss += ce_loss(y, alpha[v_num], self.classes, global_step, self.annealing_epochs)
        alpha_a = self.DS_Combin(alpha)
        evidence_a = alpha_a - 1

        loss += ce_loss(y, alpha_a, self.classes, global_step, self.annealing_epochs)
        loss = torch.mean(loss)
        return evidence, evidence_a, loss


    def collect(self, input):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """
        evidence = dict()
        for v_num in range(self.views):
            evidence[v_num] = self.Net[v_num](input[v_num])
        return evidence




class Net(nn.Module):
    def __init__(self, classifier_dims, classes):
        super(Net, self).__init__()
        self.classes = classes
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(classifier_dims[0], 64))
        self.fc.append(nn.Sigmoid())

        self.evidence = nn.ModuleList()
        self.evidence.append(nn.Linear(64, classes))
        self.evidence.append(nn.Softplus())


    def forward(self, x):

        out = self.fc[0](x)
        for i in range(1, len(self.fc)):
            out = self.fc[i](out)

        evidence = self.evidence[0](out)
        evidence = self.evidence[1](evidence)

        return evidence*evidence

