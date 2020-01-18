#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 21:20:33 2019

@author: liwei
"""

import torch
import numpy as np
def torch_Dirac_delta(x, eps):
    return 1/np.pi * eps/(x.pow(2.) + eps*eps)
# Inherit from Function
class Binarizer(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(self, input):
#        output = torch.Tensor( (input.detach().numpy()!=0).astype(int) )
        output = torch.Tensor( (input.detach().numpy()>0.0).astype(int) )
        delta_func = torch_Dirac_delta(input.double(), 1/55.)
        self.save_for_backward(delta_func)
        return output.double()

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(self, grad_output):
        output, = self.saved_tensors
        grad_input = output.mul(grad_output.double())
        return grad_input.double()

binarizer = Binarizer.apply