#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 02:09:44 2019

@author: liwei
"""
import numpy as np
import pylab as pyl
from matplotlib import pyplot as plt
#import math
import skfmm

###############################################################################
import torch
from torch.autograd import Variable
import torch.nn.functional as F

###############################################################################
#from BC_grad import BC_grad #, sign_func
from Binarizer import binarizer
###############################################################################

DEBUG = False #True
N_ITER = 2000

_res = 201
gridSize = 2./(_res-1)

dt = 0.1
kappa = 1e-5 # suggested or try 5e-5
eta = 0.1 # see the tests in the paper (0.1 to 10)

coeff = 25
mu_coeff = 2.5 #.5
mu_coeff4 = 1./32.    

###############################################################################
dtype = torch.DoubleTensor
r0 = 0.5
y_target = Variable(r0*r0*np.pi*torch.ones(1).type(dtype))
print("Target Area:", y_target.item())
###############################################################################



# For sanity check 
X, Y = np.meshgrid(np.linspace(-1-gridSize,1+gridSize,_res+2), np.linspace(-1-gridSize,1+gridSize,_res+2))

#xx, yy = np.meshgrid(np.linspace(-1,1,_res), np.linspace(-1,1,_res))

theta = np.arctan2(Y, X)
#theta = torch.from_numpy(theta)
noise1 = .01*np.cos(theta*16)
noise2 = .05*np.cos(theta*2)

#phi = -10*np.ones_like(X,dtype=np.float64)
#phi[(X)**2+(Y)**2 > 0.25] = 10
phi = np.power(X,2) + np.power(Y,2) - 0.16 + noise1 + noise2

#########################
#   Initial Condition   #
#########################
phi = skfmm.distance(phi, dx=gridSize)



#phi = Variable(phi)

kernel_dy = torch.Tensor( [[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]] ) * (1/8) / gridSize

kernel_dx = torch.Tensor( [[-1,  0,  1],
                           [-2,  0,  2],
                           [-1,  0,  1]] ) * (1/8) / gridSize

laplacian = torch.Tensor( [[ 0,  1,  0],
                           [ 1, -4,  1],
                           [ 0,  1,  0]] ) / gridSize / gridSize

partial_x = torch.Tensor([[-1, 0, 1]] ) * 0.5 / gridSize
partial_y = torch.Tensor([[-1],
                          [0],
                          [1]]) * 0.5 /gridSize



#partial_x = torch.Tensor([[0, -1, 1]] ) 
#partial_y = torch.Tensor([[0],
#                          [-1],
#                          [1]])
#laplacian = torch.Tensor( [[-1, -1, -1],
#                           [-1,  8, -1],
#                           [-1, -1, -1]] ) * (1/16)

kernel_dx  = kernel_dx.view((1,1,3,3)).type(torch.DoubleTensor)
kernel_dy  = kernel_dy.view((1,1,3,3)).type(torch.DoubleTensor)
laplacian  = laplacian.view((1,1,3,3)).type(torch.DoubleTensor)


# Initial field:
field = phi[1:-1,1:-1]

        

#phi = torch.from_numpy(phi).double()
#phi = (-torch.sigmoid(phi*coeff))+1
#print(phi.shape)
#phi = Variable(phi, requires_grad=True)

field = torch.from_numpy(field).double()
field = (-torch.sigmoid(field*coeff))+1
print(field.shape)
field = Variable(field, requires_grad=True)


pyl.figure()    
pyl.title("Initial field")
pyl.contour(X[1:-1,1:-1], Y[1:-1,1:-1], field.detach().numpy(), [0.5], colors='black', linewidths=(0.5))
pyl.pcolormesh(X[1:-1,1:-1], Y[1:-1,1:-1], field.detach().numpy())
pyl.colorbar()
pyl.gca().set_aspect(1)
pyl.xticks()
pyl.yticks()


xie = gridSize*0.5

time = []
history = []

for it in range(0,N_ITER):
    
    #phi = BC_grad(field)
    

    #binaryMask = torch.sigmoid(phi)*4.325-0.5*4.325
    binaryMask = binarizer(field - 0.5)
###############################################################################
    #phi = torch.clamp(phi, 0.5-xie, 0.5+xie)
    y_pred = binaryMask.sum()*gridSize**2
#    y_pred = phi.sum()*gridSize**2
    #y_pred = torch.clamp(phi, 0.5-xie, 0.5+xie).sum()*gridSize**2
    
    print(y_pred.item(), y_target.item())
    
    loss = (y_pred - y_target).pow(2).pow(.5)
    # sum()也是function括号不能丢, 注意在不指定dim=行/列时， 因为加变得没了方向，
    # 所以就是所有元素的加和。
    print(".....loss.....:", loss.data.item())
    loss.backward()
    
    
    
    if DEBUG:
        pyl.figure()    
        pyl.contour(X, Y, phi.detach().numpy(), [0.5], colors='black', linewidths=(0.5))
        pyl.pcolormesh(X, Y, phi.detach().numpy())
        pyl.colorbar()
        pyl.gca().set_aspect(1)
        pyl.xticks()
        pyl.yticks()
    
    
    
#    phi = phi.view(1,1,_res+2,_res+2)
    dx_layer   = F.conv2d(field.view(1,1,_res,_res), kernel_dx, padding=1).double()  #face normal vector should be 1.0 not 0.5
    dy_layer   = F.conv2d(field.view(1,1,_res,_res), kernel_dy, padding=1).double()  #face normal vector  
    dx_layer = dx_layer.view(_res, _res)
    dy_layer = dy_layer.view(_res, _res)
    
    gradient_phi = (torch.mul(dx_layer, dx_layer)+torch.mul(dy_layer, dy_layer)).pow(0.5)
    
    norm_x = torch.div(dx_layer, gradient_phi)
    norm_y = torch.div(dy_layer, gradient_phi)
    
    
    if DEBUG or it==N_ITER-1:
        pyl.figure()    
        pyl.title("Sobel Operator dphi/dx")
        pyl.pcolormesh(X[1:-1,1:-1], Y[1:-1,1:-1], dx_layer.detach().numpy())
        pyl.colorbar()
        pyl.gca().set_aspect(1)
        pyl.xticks()
        pyl.yticks()
        
        pyl.figure()    
        pyl.title("Binary Mask")
        pyl.pcolormesh(X[1:-1,1:-1], Y[1:-1,1:-1], binaryMask.detach().numpy())
        pyl.colorbar()
        pyl.gca().set_aspect(1)
        pyl.xticks()
        pyl.yticks()       
        
        plt.figure()
        pyl.title("Sobel Operator ||grad phi||")
        plt.imshow(gradient_phi.detach().numpy())
        plt.colorbar()
        
        plt.figure()
        pyl.title("Sobel Operator Norm-x")
        plt.imshow(norm_x.detach().numpy())
        plt.colorbar()
        
        plt.figure()
        theta2 = torch.atan2(dy_layer, dx_layer+1e-28)
        pyl.pcolormesh(X[1:-1,1:-1], Y[1:-1,1:-1], theta2.detach().numpy()*180./np.pi)
        pyl.colorbar()
        pyl.gca().set_aspect(1)
        pyl.xticks()
        pyl.yticks()
    
    
    term1 = F.conv2d(field.view(1,1,_res,_res), laplacian, padding=1).double() 
    term1 = term1.view(_res,_res)
    if DEBUG or it==N_ITER-1:
        plt.figure()
        plt.title("Allen-Cahn eqn. term1:")
        pyl.pcolormesh(X[1:-1,1:-1], Y[1:-1,1:-1], term1.detach().numpy())
        pyl.colorbar()
        pyl.gca().set_aspect(1)
        pyl.xticks()
        pyl.yticks()
    
    

    
    #phi = phi.view(_res+2, _res+2)
    a1 = torch.mul(field,1-field)
    a2 = field-0.5
    
    adv = field.grad.data
    #adv = torch.div(adv, torch.abs(adv + 1e-28))
    adv = torch.div(adv, adv.norm(p=2)+1e-28)
    print("Shape of d loss / d field:", adv.shape)
    a2 = a2 - 30*eta*adv
    
    a2 = torch.mul(a1, a2)



    
    if DEBUG or it==N_ITER-1:
        print("Shape of Allen-Cahn Term2:", a2.shape)
        plt.figure()
        plt.title("Allen-Cahn eqn. term2:")#    
        pyl.pcolormesh(X[1:-1,1:-1], Y[1:-1,1:-1], a2.detach().numpy())
        pyl.colorbar()
        pyl.gca().set_aspect(1)
        pyl.xticks()
        pyl.yticks()

        plt.figure()
        plt.title("d loss /d field:")#    
        pyl.pcolormesh(X[2:-2,2:-2], Y[2:-2,2:-2], field.grad.data.detach().numpy()[1:-1,1:-1])
        pyl.colorbar()
        pyl.gca().set_aspect(1)
        pyl.xticks()
        pyl.yticks()

    field.data = field.data + (kappa*term1 + a2)*dt
    field.grad.data.zero_()
    
    time.append(it)
    history.append(loss.data.item())
    
pyl.figure()  
pyl.title("Final field")  
pyl.contour(X[1:-1,1:-1], Y[1:-1,1:-1], field.detach().numpy(), [0.5], colors='black', linewidths=(0.5))
pyl.pcolormesh(X[1:-1,1:-1], Y[1:-1,1:-1], field.detach().numpy())
pyl.colorbar()
pyl.gca().set_aspect(1)
pyl.xticks()
pyl.yticks()

plt.figure()
plt.title("Loss history")
plt.plot(time,history)
plt.show()
