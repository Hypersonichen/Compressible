import numpy as np
import pylab as plt
import math
import skfmm

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from BC_grad_gridSize import BC_grad
def g_func(s, lam=1.):
    
    #s = torch.from_numpy(s)
    
    return (1+s.pow(2)).pow(-1.)
    #return (s).pow(-1.)

DEBUG = False
N_ITER = 8000

LevelSet = True
CONVECT = False
DIFFUSE = not(CONVECT)

SDF_INIT = True

SHARPEN = False #True

RE_INIT = False #True
N_SUB_ITER = 15
JST = False
###############################################################################
dtype = torch.DoubleTensor
weight = 1e-2 # 0.04 in the paper
r0 = .5
y_target = Variable(r0*r0*np.pi*torch.ones(1).type(dtype))
print("Target Area:", y_target.item())



###############################################################################

coeff = 55. #*gridSize #default is 20.
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




phi_ext = (X)**2+(Y)**2 - np.power(0.4 + noise,2.)
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
    
#    normal_x = normal_x.view(temp_idim, temp_jdim)
#    normal_y = normal_y.view(temp_idim, temp_jdim)
#
#    plt.figure()
#    plt.title("d(phi)/dx")
#    plt.pcolormesh(xx, yy, dx_layer.detach().numpy()/gridSize)
#    plt.colorbar()
#
#    
#    plt.figure()
#    plt.title("d(phi)/dy")
#    plt.pcolormesh(xx, yy, dy_layer.detach().numpy()/gridSize)
#    plt.colorbar()
    
    

    y_pred = binaryMask.sum()*gridSize**2
    print(y_pred.item())
   
    constraint = 0.5*(gradient_phi - 1.).pow(2.)
    #constraint = (gradient_phi - 1.).pow(2.).pow(.5)

    #loss = (y_pred - y_target).pow(2).pow(.5) + weight * constraint.sum()
    loss = (y_pred - y_target).pow(2).pow(.5) + weight * constraint.sum()
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
        K_const =  0.05 #1 #1. #5 #25
        tau_const = 0.75
        tau_const2 = 0 #2.5 #.5
        tau_const *= K_const
        tau_const2 *= K_const
        phi.data = phi.data - phi.grad.data * K_const  + tau_const * lap_phi * gridSize + tau_const2 * lap_sq_phi * gridSize**3
        #phi.data = phi.data - phi.grad.data * K_const  + tau_const * lap_phi + tau_const2 * lap_sq_phi
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
    
    
    if RE_INIT:
        phi_temp = phi.data    
        dtau = 1e-3
        for sub_iter in range(N_SUB_ITER):
            ###################################################
            #######   call boundary condition          ########
            ###################################################
            pMask = BC_grad(phi.data, gridSize)
            ###################################################
            
            
            temp_idim  = phi.size()[0]
            temp_jdim  = phi.size()[1]
        

            pMask = pMask.view(1, 1, temp_idim+2, temp_jdim+2)
               
             
            #lap_phi    = F.conv2d(pMask, laplacian, padding=1).double()    
            lap_phi    = F.conv2d(pMask, laplacian, padding=0).double()    
            lap_sq_phi = F.conv2d(lap_phi, laplacian, padding=1).double()
        
            lap_phi  = lap_phi.view(temp_idim, temp_jdim)
            lap_sq_phi  = lap_sq_phi.view(temp_idim, temp_jdim)
            

            if JST:
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
                
                term_2 = mu_coeff2 * lap_phi * gridSize
            else:
                dx_layer   = F.conv2d(pMask, kernel_dx, padding=0).double()  #dphi/dx #normal vector should be 1.0 not 0.5
                dy_layer   = F.conv2d(pMask, kernel_dy, padding=0).double()  #dphi/dy #face normal vector  
                gradient_phi = (torch.mul(dx_layer, dx_layer)+torch.mul(dy_layer, dy_layer)).pow(0.5)
                #gradient_phi = gradient_phi / gridSize 
        
                normal_x = torch.mul(dx_layer, gradient_phi.pow(-1.))
                normal_y = torch.mul(dy_layer, gradient_phi.pow(-1.))
        
                n_dot_dphi = torch.mul(normal_x, dx_layer) + torch.mul(normal_y, dy_layer)
                
                
                
                
                ndd_x = F.conv2d(n_dot_dphi, kernel_dx, padding=1).double()
                ndd_y = F.conv2d(n_dot_dphi, kernel_dy, padding=1).double()
                
                n_dot_dphi = torch.mul(normal_x, ndd_x) + torch.mul(normal_y, ndd_y)
                
                
        
                dx_layer = dx_layer.view(temp_idim, temp_jdim)
                dy_layer = dy_layer.view(temp_idim, temp_jdim)
                gradient_phi = gradient_phi.view(temp_idim, temp_jdim)
                n_dot_dphi = n_dot_dphi.view(temp_idim, temp_jdim)
                term_2 = mu_coeff2 * n_dot_dphi
            
            #phi.data = phi.data - dtau * torch.tanh(phi_temp*0.5*coeff)*(gradient_phi/gridSize-1.) + term_2 * dtau
    
            #phi.data = phi.data - 2 * dtau * (torch.sigmoid(phi_temp*coeff)-0.5).mul(gradient_phi-1.) #+ term_2 * dtau
            phi.data = phi.data - dtau * torch.sigmoid(coeff*torch.sigmoid(phi_temp*coeff)-0.5).mul(gradient_phi-1.) #+ term_2 * dtau
    
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
