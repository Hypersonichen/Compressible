#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 23:36:47 2019

@author: liwei
"""
import numpy as np

import torch


def sign_func(phi, h):
    return torch.div(phi, torch.pow( torch.pow(phi, 2) + h*h, 0.5))

def BC_grad(phi):
###################################################
    #######   call boundary condition          ########
    ###################################################
    bc_north  = phi[0,:]
    bc_south  = phi[-1,:]
    bc_west = phi[:,0]
    bc_east = phi[:,-1]

    bc_nw =  phi[ 0, 0]
    bc_sw =  phi[-1, 0]
    bc_ne =  phi[ 0,-1]
    bc_se =  phi[-1,-1]
    ###################################################

    bc_west = bc_west.double().view(-1,1)
    bc_east = bc_east.double().view(-1,1)
    bc_south = bc_south.double()
    bc_north = bc_north.double()

    bc_sw = torch.Tensor([bc_sw]).type(torch.DoubleTensor)
    bc_nw = torch.Tensor([bc_nw]).type(torch.DoubleTensor)
    bc_se = torch.Tensor([bc_se]).type(torch.DoubleTensor)
    bc_ne = torch.Tensor([bc_ne]).type(torch.DoubleTensor)

    bc_south = torch.cat((bc_sw,bc_south,bc_se), 0).view(1,-1)
    bc_north = torch.cat((bc_nw,bc_north,bc_ne), 0).view(1,-1)

    ###################################################

    phi_ext = torch.cat((bc_west,phi,bc_east), 1)
    #print("phi_ext shape now:",phi_ext.shape)
    phi_ext = torch.cat((bc_north,phi_ext,bc_south), 0)
    #print("phi_ext shape now:",phi_ext.shape)
    return phi_ext
