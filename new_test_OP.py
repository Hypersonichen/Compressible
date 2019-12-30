import numpy as np
import pylab as plt

import skfmm

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from BC_grad_gridSize import BC_grad
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

DEBUG = False
N_ITER = 5000

LevelSet = False
CONVECT = False
DIFFUSE = not(CONVECT)

SDF_INIT = False

SHARPEN = True

RE_INIT = False #True
N_SUB_ITER = 15
JST = False



restFile = "field.pt"
NWRITE = 20
RESTART   = False

_res = 64
gridSize = 2./(_res-1)


fsX = 0.8e-3
fsY = 0.0
_viscosity = 1e-5

upstreamVel = Velocity(fsX,fsY)
print(upstreamVel.VELPACK.VISCOSITY)
#upstreamVel.updateViscosity(1e-5)
upstreamVel.updateViscosity(_viscosity)
print("viscosity value updated: ",upstreamVel.VELPACK.VISCOSITY)


r0 = .25
velMag = fsX**2+fsY**2
dynHead = .5*velMag
velMag = math.sqrt(velMag)
print("Ref. Len. r0, Reynolds #:", 1.0*velMag*r0/_viscosity)


###############################################################################
dtype = torch.DoubleTensor

r0 = .5
y_target = Variable(r0*r0*np.pi*torch.ones(1).type(dtype))
print("Target Area:", y_target.item())



###############################################################################

coeff = 25. #*gridSize #default is 20.
mu_coeff2 = 0.001

#curv_coeff = mu_coeff2**2
mu_coeff4 = 1./128.


_res = 81

xx, yy = np.meshgrid(np.linspace(-1,1,_res), np.linspace(-1,1,_res))

gridSize = 2./(_res-1)



X, Y = np.meshgrid(np.linspace(-1-gridSize,1+gridSize,_res+2), np.linspace(-1-gridSize,1+gridSize,_res+2))
theta = np.arctan2(Y, X)
#theta = torch.from_numpy(theta)
#noise = Variable(0.1*torch.cos(theta*8).type(dtype), requires_grad=True)
#noise = .05*np.cos(theta*8)
noise = .05*np.cos(theta*8)




phi_ext = (X)**2+(Y)**2 - np.power(0.25 + noise,2.)
#phi = (X)**2+(Y)**2 - np.power(2.5, 2.)





#phi = Variable(phi)

kernel_dy = torch.Tensor( [[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]] ) * (1/8)/gridSize

kernel_dx = torch.Tensor( [[-1,  0,  1],
                           [-2,  0,  2],
                           [-1,  0,  1]] ) * (1/8)/gridSize

laplacian = torch.Tensor( [[ 0,  1,  0],
                           [ 1, -4,  1],
                           [ 0,  1,  0]] ) * (1/8)/gridSize

#laplacian = torch.Tensor( [[-1, -1, -1],
#                           [-1,  8, -1],
#                           [-1, -1, -1]] ) * (1/16)

kernel_dx  = kernel_dx.view((1,1,3,3)).type(torch.DoubleTensor)
kernel_dy  = kernel_dy.view((1,1,3,3)).type(torch.DoubleTensor)
laplacian  = laplacian.view((1,1,3,3)).type(torch.DoubleTensor)



phi_ext = skfmm.distance(phi_ext, dx=2.0/(_res-1))

phi = phi_ext[1:-1,1:-1]
phi = torch.from_numpy(phi).double()

contour_level = 0
if not(SDF_INIT):
    phi = (-torch.sigmoid(phi*coeff))+1
    contour_level = 0.5
    if SHARPEN:
        phi = torch.sigmoid((phi-0.5)*coeff)


###############################################################################

if RESTART:
    phi = torch.load("phi.pt")
else:
    phi = Variable(phi, requires_grad=True)    

#


plt.figure()
plt.title("Initial Distance")
cs = plt.contour(xx, yy, phi.detach().numpy(), [contour_level], colors='green', linewidths=(5))
plt.pcolormesh(xx, yy, phi.detach().numpy())
plt.colorbar()



time = []
history = []
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
    lap_sq_phi = F.conv2d(lap_phi, laplacian, padding=1).double()

    lap_phi  = lap_phi.view(temp_idim, temp_jdim)
    lap_sq_phi  = lap_sq_phi.view(temp_idim, temp_jdim)
    

    dx_layer   = F.conv2d(pMask, kernel_dx, padding=0).double()  #face normal vector should be 1.0 not 0.5
    dy_layer   = F.conv2d(pMask, kernel_dy, padding=0).double()  #face normal vector  
    gradient_phi = (torch.mul(dx_layer, dx_layer)+torch.mul(dy_layer, dy_layer)).pow(0.5)
    #gradient_phi = gradient_phi / gridSize 

    normal_x = torch.mul(dx_layer, gradient_phi.pow(-1.))
    normal_y = torch.mul(dy_layer, gradient_phi.pow(-1.))


    div_normal = F.conv2d(normal_x, kernel_dx, padding=1).double() + F.conv2d(normal_y, kernel_dy, padding=1).double()

    dx_layer = dx_layer.view(temp_idim, temp_jdim)
    dy_layer = dy_layer.view(temp_idim, temp_jdim)
    gradient_phi = gradient_phi.view(temp_idim, temp_jdim)
    div_normal = div_normal.view(temp_idim, temp_jdim)
    
    if DEBUG:    
        plt.figure()
        plt.title("d(phi)/dx")
        plt.imshow(dx_layer.detach().numpy())
        plt.colorbar()
        plt.show()
     
    if SDF_INIT:
        binaryMask = (-torch.sigmoid(phi*coeff))+1
    else:
        binaryMask = torch.sigmoid((phi-0.5)*coeff)
    

###################################################################################################    
    cs = plt.contour(X, Y, phi.view(_res+2, _res+2).detach().numpy(), [0.5], colors='black', linewidths=(0.5))
    p_curve = cs.collections[0].get_paths()[0]
    v = p_curve.vertices
    x_curve = v[:,0]
    y_curve = v[:,1]
    #array = np.concatenate((x_curve,y_curve),axis=1)
    writePointsIntoFile(str(epoch), v, "SQ_"+str(epoch)+".dat")
    axialChordLength = max(x_curve)-min(x_curve)
    print("Axial Chord Length: ", axialChordLength)
###################################################################################################    
##########################################################################################
# lc: copied from "Solver.py" class OpenFoamSolver def pressureSolver
    os.chdir("../data/OpenFOAM/")

    if genMesh("../../phase_field/SQ_"+str(epoch)+".dat") != 0:
        logMsg("\tmesh generation failed, aborting");
    else:
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
    ########
###############################################################
    os.chdir("../phase_field/")
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
    

    y_pred = binaryMask.sum()*gridSize**2
    print(y_pred.item())
    
    constraint = (y_pred - y_target).pow(2).pow(.5) #+ (gradient_phi.sum()/temp_idim/temp_jdim - 1.)*0.5
    loss = (drag+drag_visc)/dynHead + weight * constraint #+ lam_weight * constraint_1
    # sum()也是function括号不能丢, 注意在不指定dim=行/列时， 因为加变得没了方向，
    # 所以就是所有元素的加和。
    print(".....loss.....:", loss.data.item())
    loss.backward()
    
#####################################################################    
#####################################################################    
#####################################################################    
#    vel_max = np.amax(phi.grad.data.detach().numpy())
#    vel_min = np.amin(phi.grad.data.detach().numpy())
#    vel_max = max(abs(vel_min), abs(vel_max))
#    cfl = vel_max*dt/gridSize
#
#    dt = gridSize/vel_max*0.95
#    print("++++ CFL & vel_max,min & time-step ++++,", cfl, vel_max, vel_min, dt)

    #term_curv = curv_coeff * g_func(gradient_phi/gridSize) * div_normal/gridSize
    #term_curv = (1+(gradient_phi/gridSize).pow(2.)).pow(-1.)
#    if CONVECT: 
#        #phi.data = phi.data - phi.grad.data * dt * gradient_phi/gridSize  + dt * mu_coeff2 * lap_phi/gridSize + dt * mu_coeff4 * lap_sq_phi/gridSize
#        phi.data = phi.data - phi.grad.data * dt * gradient_phi/gridSize  
#        phi.data += (dt * mu_coeff2 * lap_phi/gridSize) #*abs(phi.grad.data) #+ dt * mu_coeff4 * lap_sq_phi/gridSize
#        #phi.data -= dt * term_curv
#    else:
    if CONVECT and LevelSet:
        K_const = 5#25
        tau_const = .5
        tau_const2 = .5
        convect = gradient_phi
        phi.data = phi.data - phi.grad.data.mul(gradient_phi) * K_const  + tau_const * lap_phi * gridSize #+ tau_const2 * lap_sq_phi
    elif DIFFUSE and LevelSet:
        K_const = 25
        tau_const = .5
        tau_const2 = .5
        phi.data = phi.data - phi.grad.data * K_const  + tau_const * lap_phi * gridSize #+ tau_const2 * lap_sq_phi
    else:
        kappa = 1e-1 # suggested or try 5e-5
        eta = .1 # see the tests in the paper 
        dt = 0.05 # learning rate
        print("pseudo time step:", dt)
        term1 = F.conv2d(pMask, laplacian, padding=0).double() 
        term1 = term1.view(_res,_res)
        
        
    
        
        #phi = phi.view(_res+2, _res+2)
        a1 = torch.mul(phi.data,1-phi.data)
        a2 = phi.data-0.5
        
        adv = phi.grad.data
        adv = torch.div(adv, adv.norm(p=2))
        #print("Shape of d loss / d field:", adv.shape)
        a2 = a2 - 30*eta*adv
        
        a2 = torch.mul(a1, a2)
        
    
    
        phi.data = phi.data + (kappa*term1 + a2)*dt
    #phi.grad.data.zero_()
    
    
    

    
    
    time.append(epoch)
    history.append(loss.data.item())
    

    if DEBUG or epoch==N_ITER-1:
        plt.figure()
        plt.title("d(loss)/d(phi)")
        plt.pcolormesh(xx, yy, phi.grad.data.detach().numpy())
        plt.colorbar()

    phi.grad.data.zero_()

plt.figure()
plt.title("Phi")
plt.contour(xx, yy, phi.detach().numpy(), [contour_level], colors='black', linewidths=(0.5))
plt.pcolormesh(xx, yy, phi.detach().numpy())
plt.colorbar()
plt.gca().set_aspect(1)
plt.xticks()
plt.yticks()

plt.figure()
plt.title("Binary Mask")
plt.contour(xx, yy, binaryMask.detach().numpy(), [0.5], colors='black', linewidths=(0.5))
plt.pcolormesh(xx, yy, binaryMask.detach().numpy())
plt.colorbar()

plt.figure()
plt.title("||grad phi||")
plt.pcolormesh(xx, yy, gradient_phi.detach().numpy())
plt.colorbar()

plt.figure()
plt.title("d(phi)/dx")
plt.pcolormesh(xx, yy, dx_layer.detach().numpy())
plt.colorbar()


plt.figure()
plt.title("d(phi)/dy")
plt.pcolormesh(xx, yy, dy_layer.detach().numpy())
plt.colorbar()
    
plt.figure()
plt.title("Loss history")
plt.plot(time,history)
plt.show()
