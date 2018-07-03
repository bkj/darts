import torch
torch.set_default_tensor_type('torch.DoubleTensor')
import numpy as np
import torch.nn as nn

from torch.autograd import Variable
from model_search import Network
from basenet.helpers import set_seeds

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])

from copy import deepcopy
from time import time
class Architect(object):

  def __init__(self, model, model_optimizer, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model_optimizer = model_optimizer
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
  
  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(
            input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid)

    grad_norm = nn.utils.clip_grad_norm(self.model.arch_parameters(), 10.)
    self.optimizer.step()
    return grad_norm

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    for v in self.model.arch_parameters():
      if v.grad is not None:
        v.grad.data.zero_()
    loss.backward()
    
  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    # _ = Network(self.model._C, self.model._num_classes, self.model._layers, self.model._criterion).cuda()
    set_seeds(987)
    
    # Backup state
    state_dict     = deepcopy(self.model.state_dict())
    opt_state_dict = deepcopy(self.model_optimizer.state_dict())
    
    # Take a step
    self.model_optimizer.zero_grad()
    train_loss = self.model._loss(input_train, target_train)
    train_loss.backward()
    self.model_optimizer.step()
    
    # Compute grad w.r.t architecture
    valid_loss     = self.model._loss(input_valid, target_valid)
    grads          = torch.autograd.grad(valid_loss, self.model.arch_parameters(), retain_graph=True)
    dtheta         = torch.autograd.grad(valid_loss, self.model.parameters())
    vector         = [dt.add(self.network_weight_decay, t).data for dt, t in zip(dtheta, self.model.parameters())]
    implicit_grads = self._hessian_vector_product(self.model, vector, input_train, target_train)
    _ = [g.data.sub_(eta, ig.data) for g, ig in zip(grads, implicit_grads)]
    
    # Reset model
    self.model.load_state_dict(state_dict)
    self.model_optimizer.load_state_dict(opt_state_dict)
    
    # Add gradient to architecture
    for v, g in zip(self.model.arch_parameters(), grads):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)
    
  def _hessian_vector_product(self, model, vector, data, target, r=1e-2):
    R = r / _concat(vector).norm()
    
    # plus R
    _ = [p.data.add_(R, v) for p, v in zip(model.parameters(), vector)]
    grads_pos = torch.autograd.grad(model._loss(data, target), model.arch_parameters())
    
    # minus R
    _ = [p.data.sub_(2*R, v) for p, v in zip(model.parameters(), vector)]
    grads_neg = torch.autograd.grad(model._loss(data, target), model.arch_parameters())
    
    return [(x - y).div_(2*R) for x, y in zip(grads_pos, grads_neg)]

