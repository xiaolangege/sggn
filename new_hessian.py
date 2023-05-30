from itertools import product
import torch
import torch.nn as nn

class Hessian_Analysis(object):
    def __init__(self, net:nn.Module):
        self.net = net
        self.idx_list = self.get_idx()

    def get_idx(self):
        idx_list = list()
        for i, params in enumerate(self.net.parameters()):
            params_idx_by_element_list = list(product(*[list(range(params.shape[dim])) for dim in range(params.ndim)]))
            idx_list += list(zip([i,]*params.numel(), params_idx_by_element_list))
        return idx_list

    def get_hessian(self, loss):
        first_grad_params = torch.autograd.grad(loss, self.net.parameters(), create_graph=True, allow_unused=True)
        f = lambda idx: self.to_flatten(
            torch.autograd.grad(first_grad_params[idx[0]][idx[1]], self.net.parameters(), retain_graph=True, allow_unused=True))
        return torch.stack(list(map(f, self.idx_list)))

    def to_flatten(self, w):
        return torch.cat([w[i].flatten() for i in range(len(w))])

