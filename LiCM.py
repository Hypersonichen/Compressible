import numpy as np
import pylab as plt

import matplotlib.pyplot as pyplt
import skfmm

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from BC_grad_gridSize import BC_grad
from Binarizer import binarizer
import shutil
import os, sys
import random
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './../')))

#--------- Project Imports ----------#

from torch.utils.data         import DataLoader
from data.utils               import makeDirs
from data.asymmetricDataGen   import *
from data.dataGen             import genMesh, runSim, processResult, outputProcessing
from train.utils              import *
from airfoil_optimizer.Helper import printTensorAsImage, logMsg
from airfoil_optimizer.DesignLoss               import * #calculateDragLift, calculateDragLift_visc
import math

_res = 64
#_res = _res+1

fsX = 8e-2
fsY = 0.0
#_viscosity = 1e-5
_viscosity = 1e-3

upstreamVel = Velocity(fsX,fsY)
print(upstreamVel.VELPACK.VISCOSITY)
#upstreamVel.updateViscosity(1e-5)
upstreamVel.updateViscosity(_viscosity)
print("viscosity value updated: ",upstreamVel.VELPACK.VISCOSITY)
#


#dtype = torch.FloatTensor
dtype = torch.DoubleTensor

r0 = 0.25
y_target = Variable(r0*r0*np.pi*torch.ones(1).type(dtype))
print("Target Area:", y_target.item())
velMag = fsX**2+fsY**2
dynHead = .5*velMag
velMag = math.sqrt(velMag)
print("Ref. Len. r0= ", r0, "; Reynolds #=", 1.0*velMag*r0/_viscosity)

DEBUG_IMAGES = False
restFile = "phi.pt"
NWRITE = 2
NRESET = 20
RESTART   = False #True
DYN_DT    = True #False
WITH_NOISE= False #True
WITH_SHAPE_TEST= False
##################False ##############################################################
N_ITER = 20




X, Y = np.meshgrid(np.linspace(-1,1,_res), np.linspace(-1,1,_res))

gridSize = 2./(_res-1)

theta = np.arctan2(Y, X)
#theta = torch.from_numpy(theta)
#noise = Variable(0.1*torch.cos(theta*8).type(dtype), requires_grad=True)
noise = .01*np.cos(theta*8)*r0
noise1 = .25*np.cos(theta*1 + np.pi*0.5)*r0*2
noise2 = 0.2*np.cos(theta*1)


if WITH_SHAPE_TEST:
    phi = (X)**2+(Y)**2/0.25 - np.power(r0 + noise1 + noise2,2.)
elif WITH_NOISE: 
    phi = (X)**2+(Y)**2 - np.power(r0 + noise,2.)
else:
    #phi = (X)**2+(Y)**2/0.16 - np.power(r0,2.) # for rugby shape test (r0=.5)
    phi = (X)**2+(Y)**2 - np.power(r0,2.)


dt = 0.025 # learning rate
#dt = 0.1 # learning rate
#dt = 0.01 # learning rate
#dt = 5 # learning rate
print("pseudo time step:", dt)


#phi = Variable(phi)

kernel_dx = torch.Tensor( [[-1,  0,  1],
                           [-2,  0,  2],
                           [-1,  0,  1]] ) * (1/8) / gridSize
kernel_dy = torch.Tensor( [[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]] ) * (1/8) / gridSize

#
#kernel_dy = torch.Tensor( [[ 0, -1,  0],
#                           [ 0,  0,  0],
#                           [ 0,  1,  0]] ) * (1/2) / gridSize
#
#kernel_dx = torch.Tensor( [[ 0,  0,  0],
#                           [-1,  0,  1],
#                           [ 0,  0,  0]] ) * (1/2) / gridSize

laplacian = torch.Tensor( [[ 0,  1,  0],
                           [ 1, -4,  1],
                           [ 0,  1,  0]] ) * (1/8) / gridSize / gridSize
#
#laplacian = torch.Tensor( [[-1, -1, -1],
#                           [-1,  8, -1],
#                           [-1, -1, -1]] ) * (1/16)

kernel_dx  = kernel_dx.view((1,1,3,3)).type(torch.DoubleTensor)
kernel_dy  = kernel_dy.view((1,1,3,3)).type(torch.DoubleTensor)
laplacian  = laplacian.view((1,1,3,3)).type(torch.DoubleTensor)


###############################################################################

if RESTART:
    phi = torch.load("phi.pt")
else:
    #phi = skfmm.distance(phi, dx=2.0/100)
    phi = skfmm.distance(phi, dx=2.0/(_res-1))
    phi = torch.from_numpy(phi).double()
#phi=phi.t()
#phi.requires_grad_()

###############################################################################

phi = Variable(phi, requires_grad=True)    

#w_lam = torch.ones(1).type(dtype)
#w_lam = torch.nn.Parameter(w_lam, requires_grad=True)
#w_lam2 = torch.ones(1).type(dtype)
#w_lam2 = torch.nn.Parameter(w_lam2, requires_grad=True)
#w_lam3 = torch.ones(1).type(dtype)
#w_lam3 = torch.nn.Parameter(w_lam3, requires_grad=True)
#
coeff =55. # 500.*gridSize #default is 20.

#mu_coeff2 = 5. #2.25*gridSize ~ 0.07
mu_coeff2 = 2.5
#mu_coeff2 = 0.25

#mu_coeff_grad = 0.01

plt.figure()
plt.title("Initial Distance")
cs = plt.contour(X, Y, phi.detach().numpy(), [0], colors='green', linewidths=(5))
plt.pcolormesh(X, Y, phi.detach().numpy())
plt.colorbar()



plt.figure()
plt.title("Distance")


#grad_weight = 0.01 #0.025 #0.01 # 0.04 suggested in the paper
weight_init = 50 #15000
weight_max = 50
lam_weight_init = 0 #-10.603073468229347  #-21.5 #-384.96444224072087# 0

weight = weight_init
lam_weight = lam_weight_init

weight2 = 10 #2.
weight3 = 10 #50.


plt.figure()
plt.title("phi in image space")
plt.imshow(phi.detach().numpy())

#phi = phi.t()
##################
for epoch in range(N_ITER):
    
    
    #plt.figure()

    #plt.title("Distance at Epoch:"+str(epoch))
    ###################################################
    #######   call boundary condition          ########
    ###################################################
    pMask = BC_grad(phi,gridSize)
    
    ###################################################
    
    
    temp_idim  = phi.size()[0]
    temp_jdim  = phi.size()[1]

    pMask = pMask.view(1, 1, temp_idim+2, temp_jdim+2)
       
     
    #lap_phi    = F.conv2d(pMask, laplacian, padding=1).double()    
    lap_phi    = F.conv2d(pMask, laplacian, padding=0).double()    
    #lap_sq_phi = F.conv2d(lap_phi, laplacian, padding=1).double()
    
    lap_phi  = lap_phi.view(temp_idim, temp_jdim)
    #lap_sq_phi  = lap_sq_phi.view(temp_idim, temp_jdim)
    #plt.figure()
    #plt.title("lap_phi")
    #plt.pcolormesh(X, Y, lap_phi.detach().numpy())
    #plt.colorbar()
    if DEBUG_IMAGES:
        pyplt.figure()
        pyplt.title("Extended phi in pixel space")
        pyplt.imshow(pMask.view(temp_idim+2, temp_jdim+2).detach().numpy())
        pyplt.colorbar() 
    if True:
        dx_layer   = F.conv2d(pMask, kernel_dx, padding=0).double()  #face normal vector should be 1.0 not 0.5
        dy_layer   = F.conv2d(pMask, kernel_dy, padding=0).double()  #face normal vector  
        #dx_layer = dx_layer.view(temp_idim, temp_jdim)
        #dy_layer = dy_layer.view(temp_idim, temp_jdim)

        gradient_phi = (torch.mul(dx_layer, dx_layer)+torch.mul(dy_layer, dy_layer)).pow(0.5)    
        normal_x = torch.div(dx_layer, gradient_phi)
        normal_y = torch.div(dy_layer, gradient_phi)


        div_normal = F.conv2d(normal_x, kernel_dx, padding=1).double() + F.conv2d(normal_y, kernel_dy, padding=1).double()
       
        dx_layer = dx_layer.view(temp_idim, temp_jdim)
        dy_layer = dy_layer.view(temp_idim, temp_jdim)
        gradient_phi = gradient_phi.view(temp_idim, temp_jdim)
        div_normal = div_normal.view(temp_idim, temp_jdim) 
       
    binaryMask = (-torch.sigmoid(phi*coeff))+1
    #binaryMask = binarizer(-phi)
    #binaryMask = torch.sigmoid((binaryMask-0.5)*coeff)
    
    if DEBUG_IMAGES:
        pyplt.figure()
        pyplt.title("Binary mask in pixel space")
        pyplt.imshow(binaryMask.view(temp_idim, temp_jdim).detach().numpy())
        pyplt.colorbar()
###################################################################################################    
    cs = plt.contour(X, Y, phi.detach().numpy(), [0], colors='black', linewidths=(0.5))
    p_curve = cs.collections[0].get_paths()[0]
    v = p_curve.vertices
    x_curve = v[:,0]
    y_curve = v[:,1]
    #array = np.concatenate((x_curve,y_curve),axis=1)
    writePointsIntoFile(str(epoch), v, "SQ_"+str(epoch)+".dat")
    axialChordLength = max(x_curve)-min(x_curve)
    verticalHeight = max(y_curve)-min(y_curve)
    print("Axial Chord Length: ", axialChordLength, "Asp. Ratio", axialChordLength/verticalHeight)
###################################################################################################    
##########################################################################################
# lc: copied from "Solver.py" class OpenFoamSolver def pressureSolver
    os.chdir("../data/OpenFOAM/")
    
    if genMesh("../../cylinder_optimizer/SQ_"+str(epoch)+".dat") != 0:
        logMsg("\tmesh generation failed, aborting");
    else:
        #print( "something")
        runSim(fsX, fsY, _res, 1)
    
    os.chdir("..")
##########################################################################################
##########################################################################################
    coordX = outputProcessing('OptSim_' + str(epoch), fsX, fsY, 
                                res=_res, binaryMask=None, 
                                oversamplingRate=1, imageIndex=epoch)[0]
    coordY = outputProcessing('OptSim_' + str(epoch), fsX, fsY, 
                                res=_res, binaryMask=None, 
                                oversamplingRate=1, imageIndex=epoch)[1]
    #binaryMask = outputProcessing('OptSim_' + str(epoch), fsX, fsY, 
    #                            res=res, binaryMask=None, 
    #                            oversamplingRate=1, imageIndex=epoch)[2]
    ##binaryMask.tofile("./temp_binaryMask.dat",sep=" ",format="%s")
    pressure = outputProcessing('OptSim_' + str(epoch), fsX, fsY, 
                                res=_res, binaryMask=None, 
                                oversamplingRate=1, imageIndex=epoch)[3]
    #pressure.tofile("./temp_pressure.dat",sep=" ",format="%s")
    #lc##### 
    velocityX = outputProcessing('OptSim_' + str(epoch), fsX, fsY, 
                                res=_res, binaryMask=None, 
                                oversamplingRate=1, imageIndex=epoch)[4]
    velocityY = outputProcessing('OptSim_' + str(epoch), fsX, fsY, 
                                res=_res, binaryMask=None, 
                                oversamplingRate=1, imageIndex=epoch)[5]
    ########
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #pressure = np.flipud(pressure.transpose())
    ##lc##### 
    #velocityX = np.flipud(velocityX.transpose())
    #velocityY = np.flipud(velocityY.transpose())
    pressure = pressure.transpose()
    #lc##### 
    velocityX = velocityX.transpose()
    velocityY = velocityY.transpose()
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if DEBUG_IMAGES:
        pyplt.figure()
        pyplt.imshow(velocityX)
        pyplt.colorbar()
        plt.figure()
        plt.title("velocityX in the physical space")
        plt.pcolormesh(X, Y, velocityX)
        plt.colorbar()
        #plt.show()
        plt.figure()
        plt.imshow(pressure)
        plt.colorbar()
        plt.figure()
        plt.imshow(X)
        plt.colorbar()
        plt.figure()
        plt.imshow(Y)
        plt.colorbar()
        plt.figure()
        plt.imshow(phi.detach().numpy())
        plt.colorbar()
    ########
    ########
###############################################################
    os.chdir("../cylinder_optimizer/")
#torch.from_numpy(pressure.copy()).double(), torch.from_numpy(velocityX.copy()).double(), torch.from_numpy(velocityY.copy()).double()
    coordX = torch.from_numpy(coordX.copy()).type(dtype)
    coordY = torch.from_numpy(coordY.copy()).type(dtype)
    #binaryMask = torch.from_numpy(binaryMask.copy()).type(dtype)
    pressure   = torch.from_numpy(pressure.copy()).type(dtype)
    velocityX  = torch.from_numpy(velocityX.copy()).type(dtype)
### BE CAREFUL !!!!!  no need to reverse sign
    velocityY  = torch.from_numpy(velocityY.copy()).type(dtype)
### BE CAREFUL !!!!!
#    utils.saveAsImage('coordX.png', coordX.view(res,res).detach().numpy()) # [2] binary mask for boundary
#    utils.saveAsImage('coordY.png', coordY.view(res,res).detach().numpy()) # [2] binary mask for boundary


    #Criterion = DragLoss(velx=_VELX, vely=_VELY, logStates=_VERBOSE[2], verbose=False, solver=PressureSolver.pressureSolver)
    drag, lift = calculateDragLift(binaryMask, pressure, upstreamVel.passTupple(), False, _res)
    drag_visc, lift_visc = calculateDragLift_visc(binaryMask, velocityX, velocityY, upstreamVel.passTupple(), False)
    print(drag.item(), drag_visc.item())
    print(lift.item(), lift_visc.item())
    
    
#    plt.gca().set_aspect(1)
    
    
    y_pred = binaryMask.sum()*gridSize**2
    xc_pred = torch.mul(binaryMask, torch.from_numpy(X).double())
    xc_pred = xc_pred.sum() * binaryMask.sum().pow(-1.)
    yc_pred = torch.mul(binaryMask, torch.from_numpy(Y).double())
    yc_pred = yc_pred.sum() * binaryMask.sum().pow(-1.)
    print("Pred. & Target Areas:", y_pred.item(), y_target.item())
    print("pred. xc & yc:", xc_pred.item(), yc_pred.item()) 
    weight = weight * math.sqrt(10)
    weight = min(weight, weight_max)
    print("-+-+-+-+ update weight", weight)

    weight2 = weight2*1.
    weight3 = weight3 * 1.00

    #grad_constraint = 0.5*(gradient_phi - 1.).pow(2.)
    #grad_constraint = (gradient_phi - 1.).pow(2.)
    #grad_constraint_1 = (gradient_phi - 1.)
    #constraint = (gradient_phi - 1.).pow(2.).pow(.5)
    #grad_constraint = grad_constraint.mul(1-binaryMask)
    #loss = (y_pred - y_target).pow(2).pow(.5) + weight * constraint.sum()

    #constraint = (y_pred - y_target).pow(2.).pow(.5)      # Form - 1

    constraint = (y_pred - y_target).pow(2.)               # Form - 2
    constraint_1 = (y_pred - y_target)                     # Form - 2

    #loss = (drag + drag_visc).pow(2).pow(0.5)/dynHead + weight * w_lam * constraint  
    
    #constraint_2  = (lift + lift_visc).pow(2)/dynHead/dynHead  
    #constraint_2 = torch.abs(lift + lift_visc)/dynHead 
    
    #constraint_3  = (xc_pred.pow(2) + yc_pred.pow(2)).pow(0.5)
    #constraint_3b = (xc_pred.pow(2) + yc_pred.pow(2)).pow(0.5)

    loss = (drag+drag_visc)/dynHead + weight * constraint + lam_weight * constraint_1
    #loss = (drag)/dynHead + weight * constraint #+ lam_weight * constraint_1 #################################### change here!
    #loss+= weight2 * constraint_2 + weight3 * constraint_3
    #loss += grad_weight * grad_constraint.sum() * gridSize*gridSize #* weight ###########################change here!
    #loss += weight * grad_weight * grad_constraint.sum() 
    #loss += lam_weight * grad_weight * grad_constraint_1.sum() 

    # sum()也是function括号不能丢, 注意在不指定dim=行/列时， 因为加变得没了方向，
    # 所以就是所有元素的加和。
#    loss = drag_visc
    print(".....loss.....:", loss.data.item(), drag.item()/dynHead, drag_visc.item()/dynHead)
    loss.backward()
    
    #print(torch.sign(phi).shape) #this would work
   
#####################################################################    
#####################################################################    
#####################################################################    
    vel_max = np.amax(phi.grad.data.detach().numpy())
    vel_min = np.amin(phi.grad.data.detach().numpy())
    vel_max = max(abs(vel_min), abs(vel_max))
    cfl = vel_max*dt/gridSize

    if DYN_DT:
        dt = gridSize/vel_max*0.5

    print("++++ CFL & vel_max,min & time-step ++++,", cfl, vel_max, vel_min, dt)
#####################################################################    
#####################################################################    
#####################################################################    
    term_1 = - phi.grad.data #* gradient_phi
    term_2 =  mu_coeff2 * lap_phi * gridSize * abs(phi.grad.data)
    term_grad = 0 #mu_coeff_grad * (lap_phi - div_normal)
    #phi.data = phi.data + (term_1 + term_2 + term_grad)*dt 
    phi.data = phi.data + (term_1 + term_2)*dt 


    binaryMask = (-torch.sigmoid(phi*coeff))+1
    #binaryMask = binarizer(-phi)
    #binaryMask = torch.sigmoid((binaryMask-0.5)*coeff)
    y_pred = binaryMask.sum()*gridSize**2
    constraint_1 = (y_pred - y_target)                     # Form - 2
    lam_weight = lam_weight + 2*weight*constraint_1.item()
    print("-+-+-+-+ update lam_weight", lam_weight)






    if DEBUG_IMAGES:
        plt.figure()
        plt.title("phi")
        plt.pcolormesh(X, Y, phi.detach().numpy())
        plt.colorbar()
        
        plt.figure()
        plt.title("dLoss/dphi")
        plt.pcolormesh(X, Y, phi.grad.data.detach().numpy())
        plt.colorbar()

        plt.figure()
        plt.title("term1")
        plt.pcolormesh(X, Y, term_1.detach().numpy())
        plt.colorbar()

        plt.figure()
        plt.title("term2")
        plt.pcolormesh(X, Y, term_2.detach().numpy())
        plt.colorbar()

        plt.figure()
        plt.title("term_grad")
        plt.pcolormesh(X, Y, term_grad.detach().numpy())
        plt.colorbar()

        plt.figure()
        plt.title("||grad(phi)||")
        plt.pcolormesh(X, Y, gradient_phi.detach().numpy())
        plt.colorbar()
        plt.show()
#####################################################################    
#####################################################################    
#####################################################################    
    phi.grad.data.zero_()
    
#    w_lam.data -= dt * w_lam.grad.data
#    w_lam.grad.data.zero_()
    
#    w_lam2.data -= dt * w_lam2.grad.data
#    w_lam2.grad.data.zero_()
    
#
#    w_lam3.data -= dt * w_lam3.grad.data
#    w_lam3.grad.data.zero_()

    if epoch>0 and epoch%NWRITE==0:
        #np.save(restFile, phi.detach.numpy())
        torch.save(phi, restFile) 
        print("Save phi at Epoch", epoch)
    if epoch>0 and epoch%NRESET==0:
        weight = weight_init #math.sqrt(10)
        lam_weight = 0
        print("Reset weights at Epoch", epoch)
    
    
    
    
plt.figure()
plt.contour(X, Y, phi.detach().numpy(), [0], colors='black', linewidths=(0.5))
plt.pcolormesh(X, Y, phi.detach().numpy())
plt.colorbar()







plt.gca().set_aspect(1)
plt.xticks()
plt.yticks()
    
plt.show()
