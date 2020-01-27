import numpy as np
import pylab as plt

import matplotlib.pyplot as pyplt
from mpl_toolkits import mplot3d
import skfmm

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from BC_grad_gridSize import BC_grad
from Binarizer import binarizer
def Heaviside(x, eps):
    return 0.5 + 1/np.pi * np.arctan2(x, eps)
def Dirac_delta(x, eps):
    return 1/np.pi * eps / (np.power(x,2) + eps*eps)

def torch_Heaviside(x, eps):
#    eps = torch.from_numpy(eps).double()
    eps = eps.expand(x.size()[0], x.size()[1])
    return 0.5 + 1/np.pi * torch.atan2(x, eps)
def torch_Dirac_delta(x, eps):
    return 1/np.pi * eps/(x.pow(2.) + eps*eps)

_res = 128
#_res = _res+1
#


#dtype = torch.FloatTensor
dtype = torch.DoubleTensor

r0 = 0.4
y_target = Variable(r0*r0*np.pi*torch.ones(1).type(dtype))

DEBUG_IMAGES = False
restFile = "phi.pt"
NWRITE = 1000

RESTART   = False
DYN_DT    = True #False
WITH_NOISE= False #True
WITH_SHAPE_TEST= True #False
##################False ##############################################################
N_ITER = 100




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
    phi = (X)**2+(Y)**2 - np.power(r0/2. + noise,2.)
else:
    #phi = (X)**2+(Y)**2/0.16 - np.power(r0,2.) # for rugby shape test (r0=.5)
    phi = (X)**2+(Y)**2 - np.power(r0*0.5,2.)

#eps = 0.0005
dt = 1e-3 # learning rate
#dt = 0.1 # learning rate
#dt = 0.01 # learning rate
#dt = 5 # learning rate
print("pseudo time step:", dt)


#phi = Variable(phi)

#kernel_dx = torch.Tensor( [[-1,  0,  1],
#                           [-2,  0,  2],
#                           [-1,  0,  1]] ) * (1/8) / gridSize
#kernel_dy = torch.Tensor( [[-1, -2, -1],
#                           [ 0,  0,  0],
#                           [ 1,  2,  1]] ) * (1/8) / gridSize


kernel_dy = torch.Tensor( [[ 0, -1,  0],
                           [ 0,  0,  0],
                           [ 0,  1,  0]] ) * (1/2) / gridSize

kernel_dx = torch.Tensor( [[ 0,  0,  0],
                           [-1,  0,  1],
                           [ 0,  0,  0]] ) * (1/2) / gridSize

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
#mu_coeff2 = 2.5
mu_coeff2 = 0.0005 #0.8 #0.2 #0.1 #0.25
mu_coeff_grad = 0.000625 #1e-5 #1e-3

plt.figure()
plt.title("Initial Distance")
cs = plt.contour(X, Y, phi.detach().numpy(), [0], colors='green', linewidths=(5))
plt.pcolormesh(X, Y, phi.detach().numpy())
plt.colorbar()


lam_grad = torch.Tensor([[0]]).double()
lam_grad = lam_grad.expand(_res,_res)



plt.figure()
plt.title("phi in image space")
plt.imshow(phi.detach().numpy())

time = []
history = []




# Defines a SGD optimizer to update the parameters
#optimizer = optim.SGD([phi], lr=dt/eps)
optimizer = optim.Adam([phi], lr=dt, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
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



    dx_layer   = F.conv2d(pMask, kernel_dx, padding=0).double()  #face normal vector should be 1.0 not 0.5
    dy_layer   = F.conv2d(pMask, kernel_dy, padding=0).double()  #face normal vector  
    #dx_layer = dx_layer.view(temp_idim, temp_jdim)
    #dy_layer = dy_layer.view(temp_idim, temp_jdim)

    gradient_phi = (torch.mul(dx_layer, dx_layer)+torch.mul(dy_layer, dy_layer)).pow(0.5)    

    
    
#    diff_rate = 1 - gradient_phi.pow(-1.)
    diff_rate = torch_Heaviside(gradient_phi - 1., torch.Tensor([1./55.]).double())
    normal_x = diff_rate.mul(dx_layer)
    normal_y = diff_rate.mul(dy_layer)
    div_normal = F.conv2d(normal_x, kernel_dx, padding=1).double() + F.conv2d(normal_y, kernel_dy, padding=1).double()
    

    

    normal_x = torch.div(dx_layer, gradient_phi)
    normal_y = torch.div(dy_layer, gradient_phi)
#    div_normal = F.conv2d(normal_x, kernel_dx, padding=1).double() + F.conv2d(normal_y, kernel_dy, padding=1).double()
#    div_normal = lap_phi - div_normal    
    
    
    
    dx_layer = dx_layer.view(temp_idim, temp_jdim)
    dy_layer = dy_layer.view(temp_idim, temp_jdim)
    gradient_phi = gradient_phi.view(temp_idim, temp_jdim)
    div_normal = div_normal.view(temp_idim, temp_jdim) 
    normal_x = normal_x.view(temp_idim, temp_jdim)
    normal_y = normal_y.view(temp_idim, temp_jdim)
    
    
    
       
    #binaryMask = (-torch.sigmoid(phi*coeff))+1
    DiracMask  = torch_Dirac_delta(-phi, 1/coeff)
    binaryMask = binarizer(-phi)
    #binaryMask = torch.sigmoid((binaryMask-0.5)*coeff)
    



    y_pred = binaryMask.sum()*gridSize**2 
    
    constraint_grad = 0.5*(gradient_phi - 1.).pow(2.).sum()*gridSize**2
#    constraint_1 = gradient_phi.mul(1-binaryMask) - 1.
    constraint = .5*(y_pred - y_target).pow(2.)


    loss =  constraint + mu_coeff_grad * constraint_grad
    #+ lam_grad.mul(constraint_1).sum()
    print("Pred. & Target Areas:", y_pred.item(), y_target.item())
    #loss = (drag)/dynHead + weight * constraint #+ lam_weight * constraint_1 #################################### change here!
    loss.backward()
    
    #print(torch.sign(phi).shape) #this would work
   
#####################################################################    
#####################################################################    
#####################################################################   
##    term_1 = - phi.grad.data.mul( gradient_phi )
##    term_2 =  mu_coeff2 * lap_phi * gridSize * abs(phi.grad.data)
#    
#
#    #term_1 = phi.grad.data.mul( DiracMask ) / eps
#    term_1 = -phi.grad.data / eps
    term_2 = mu_coeff2 * lap_phi
##    phi.data = phi.data + (term_1 + term_2)*dt 
#
#    term_grad = mu_coeff_grad * div_normal
#    
#    #phi.data = phi.data + (term_1 + term_2 + term_grad)*dt 
#    
    optimizer.step()
    phi.data += term_2*dt
    
    # No more telling PyTorch to let gradients go!
    # a.grad.zero_()
    # b.grad.zero_()
    optimizer.zero_grad()



#    pMask = BC_grad(phi,gridSize)
#    
#    ###################################################
#    
#    
#    temp_idim  = phi.size()[0]
#    temp_jdim  = phi.size()[1]
#
#    pMask = pMask.view(1, 1, temp_idim+2, temp_jdim+2)
#       
#
#    dx_layer   = F.conv2d(pMask, kernel_dx, padding=0).double()  #face normal vector should be 1.0 not 0.5
#    dy_layer   = F.conv2d(pMask, kernel_dy, padding=0).double()  #face normal vector  
#
#    gradient_phi = (torch.mul(dx_layer, dx_layer)+torch.mul(dy_layer, dy_layer)).pow(0.5)    
#  
#    dx_layer = dx_layer.view(temp_idim, temp_jdim)
#    dy_layer = dy_layer.view(temp_idim, temp_jdim)
#    gradient_phi = gradient_phi.view(temp_idim, temp_jdim)
#
#
#    lam_grad.data = lam_grad.data + mu_coeff_grad*(gradient_phi -1)


    if DEBUG_IMAGES or epoch==N_ITER-1:
        plt.figure()
        plt.title("phi")
        plt.pcolormesh(X, Y, phi.detach().numpy())
        plt.colorbar()
        
        plt.figure()
        plt.title("dLoss/dphi")
        plt.contour(X, Y, phi.detach().numpy(), [0], colors='black', linewidths=(0.5))
        plt.pcolormesh(X, Y, phi.grad.data.detach().numpy())
        plt.colorbar()
        
        plt.figure()
        plt.title("binaryMask")
        plt.contour(X, Y, phi.detach().numpy(), [0], colors='black', linewidths=(0.5))
        plt.pcolormesh(X, Y, binaryMask.detach().numpy())
        plt.colorbar()
        
        plt.figure()
        plt.title("DiracMask")
        plt.contour(X, Y, phi.detach().numpy(), [0], colors='black', linewidths=(0.5))
        plt.pcolormesh(X, Y, DiracMask.detach().numpy())
        plt.colorbar()
#
#        plt.figure()
#        plt.title("term1")
#        plt.contour(X, Y, phi.detach().numpy(), [0], colors='black', linewidths=(0.5))
#        plt.pcolormesh(X, Y, term_1.detach().numpy())
#        plt.colorbar()
#
#        plt.figure()
#        plt.title("term2")
#        plt.contour(X, Y, phi.detach().numpy(), [0], colors='black', linewidths=(0.5))
#        plt.pcolormesh(X, Y, term_2.detach().numpy())
#        plt.colorbar()

#        plt.figure()
#        plt.title("term_grad")
#        plt.pcolormesh(X, Y, term_grad.detach().numpy())
#        plt.colorbar()
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, phi.detach().numpy(), rstride=1, cstride=1, cmap='viridis', edgecolor='none');
        
        plt.figure()
        plt.title("norm_x")
        plt.contour(X, Y, phi.detach().numpy(), [0], colors='black', linewidths=(0.5))
        plt.pcolormesh(X, Y, normal_x.detach().numpy())
        plt.colorbar()


        plt.figure()
        plt.title("||grad(phi)||")
        plt.pcolormesh(X, Y, gradient_phi.detach().numpy())
        plt.colorbar()
        
        plt.figure()
        plt.title("||grad(phi)||")
        plt.imshow(gradient_phi.detach().numpy())
        plt.colorbar()
        plt.show()
#####################################################################    
#####################################################################    
#####################################################################    
    phi.grad.data.zero_()
    

    
    time.append(epoch)
    history.append(loss.data.item())
    
    if epoch>0 and epoch%NWRITE==0:
        #np.save(restFile, phi.detach.numpy())
        torch.save(phi, restFile)
        print("Save phi at Epoch", epoch)
    
plt.figure()
plt.contour(X, Y, phi.detach().numpy(), [0], colors='black', linewidths=(0.5))
plt.pcolormesh(X, Y, phi.detach().numpy())
plt.colorbar()




plt.figure()
plt.title("Loss history")
plt.plot(time,history)
plt.show()



