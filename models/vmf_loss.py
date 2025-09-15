import torch
import torch.nn as nn
import torch.nn.functional as Fn
import numpy as np
import scipy
from scipy.special import ive
from numbers import Number
import math


def compute_mu(x):
    r = torch.sum(x, dim=0)
    mu = r / torch.norm(r)
    mu = mu.view(-1, 1)
    return mu

def compute_kappa(x):
    r = torch.sum(x, dim=0)
    norm_r = torch.norm(r)
    n = x.shape[0]
    d = x.shape[1]
    r_bar = norm_r / n

    kappa = (d * r_bar - r_bar**3) / (1 - r_bar**2)
    if r_bar > 0.9:
        kappa = -0.4 + 1.39 * r_bar + 0.43 / (1 - r_bar)
    return kappa

class IveFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, z):
        assert isinstance(v, Number), "v must be a scalar"
        ctx.save_for_backward(z)
        ctx.v = v
        z_cpu = z.data.cpu().numpy()

        if np.isclose(v, 0):
            output = scipy.special.i0e(z_cpu, dtype=z_cpu.dtype)
        elif np.isclose(v, 1):
            output = scipy.special.i1e(z_cpu, dtype=z_cpu.dtype)
        else:
            output = scipy.special.ive(v, z_cpu, dtype=z_cpu.dtype)
        output = torch.from_numpy(np.array(output)).to(z.device)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        z = ctx.saved_tensors[-1]
        v = ctx.v
        return (None, grad_output * (ive(v - 1, z) - ive(v, z) * (v + z) / z))

class Ive(nn.Module):
    def __init__(self, v):
        super(Ive, self).__init__()
        self.v = v

    def forward(self, z):
        return ive(self.v, z)

ive = IveFunction.apply

def kl_divergence_vmf(mu_q, mu_p, kappa_q, kappa_p):
    d = mu_q.shape[0]
    if d % 2 == 0:
        d += 1
    d_star = (d / 2) - 1

    ive_module = Ive(d_star).to(mu_q.device)
    bes_kq = ive_module(kappa_q)
    bes_kp = ive_module(kappa_p)

    step1 = d_star * torch.log(kappa_p)
    step2 = d_star * torch.log(kappa_q)
    step3 = bes_kq
    step4 = bes_kp
    step5 = kappa_p * torch.matmul(mu_p.t(), mu_q)

    result = step1 - step2 - kappa_q + step3 - step4 + step5
    return result

def compute_vMF(X, y):
    labels = torch.unique(y)
    mu_list = []
    kappa_list = []

    for k in labels:
        X_class = X[y == k,:]
        mu = compute_mu(X_class)
        kappa = compute_kappa(X_class)
        mu_list.append(mu)
        kappa_list.append(kappa.unsqueeze(0))

    mu_list = torch.cat(mu_list, dim=1)
    kappa_list = torch.cat(kappa_list, dim=0)

    return mu_list.transpose(0, 1), kappa_list

def compute_pairwise_kl_divergence(mu_list, kappa_list):
    K, D = mu_list.shape
    kl_matrix = torch.zeros(K, K, device=mu_list.device)

    for i in range(K):
        for j in range(K):
            kl_matrix[i, j] = kl_divergence_vmf(mu_list[i, :], mu_list[j, :], kappa_list[i], kappa_list[j])

    return kl_matrix

class VMFLoss(nn.Module):
    def __init__(self):
        super(VMFLoss, self).__init__()
    def forward(self, X, y):
        X = X.float()
        y = y.float()

        mu_list, kappa_list = compute_vMF(X, y)
        kl_matrix = compute_pairwise_kl_divergence(mu_list, kappa_list)
        kl_loss = torch.sum(kl_matrix ** 2) / (kl_matrix.shape[0] ** 2)

        return kl_loss
    
import torch.nn as nn

def log_prob(x,scale, loc, d):
        return log_unnormalized_prob(x,scale,loc) - log_normalization(scale, d)

def log_unnormalized_prob(x,scale, loc):

    output = scale * (loc * x).sum(-1, keepdim=True)

    return output.squeeze()

def log_normalization(scale, d):

    term1 = (d / 2 - 1) * torch.log(scale)
    term2 = (d / 2) * math.log(2 * math.pi)
    term3 = scale + torch.log(ive(d/ 2 - 1, scale)+1e-8)

    output = -(term1 - term2 - term3)

    return output

def log_likelihood(X, mu_list, kappa_list, amp = 2):
    N, D = X.shape
    K = len(mu_list)

    loglikehood_matrix = torch.zeros(N, K)

    kappa_list = (kappa_list.detach())/amp
    kappa_list = kappa_list.detach()

    for i in range(K):
        loglikehood_matrix[:, i] = log_prob(X, kappa_list[i], mu_list[i], D).squeeze()

    return loglikehood_matrix



def normalize(loglikehood_matrix):
    return Fn.softmax(loglikehood_matrix, dim=1)

def vMF_centroid_loss(X,y,amp=2, amp2=2):
    mu_list, kappa_list = compute_vMF(X, y)

    loglikehood_matrix = log_likelihood(X, mu_list, kappa_list, amp = amp)
    loglikehood_matrix = loglikehood_matrix.to(X.device)

    normalized_loglikehood_matrix = normalize(amp2*loglikehood_matrix+1e-8)
    normalized_loglikehood_matrix = normalized_loglikehood_matrix.to(X.device)

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(normalized_loglikehood_matrix, y)

    return loss