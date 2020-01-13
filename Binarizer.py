#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 21:20:33 2019

@author: liwei
"""

import torch
import numpy as np

# Inherit from Function
class Binarizer(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(self, input):
#        output = torch.Tensor( (input.detach().numpy()!=0).astype(int) )
        output = torch.Tensor( (input.detach().numpy()>0.0).astype(int) )
        self.save_for_backward(output.double())
        return output.double()

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(self, grad_output):
        output, = self.saved_tensors
        grad_input = output.mul(grad_output.double())
        return grad_input.double()

binarizer = Binarizer.apply