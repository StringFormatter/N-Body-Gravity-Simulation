import math
import numpy as np
from numpy import fft

def DensityField(points, Nc, L):
    density = np.zeros((Nc, Nc, Nc))
    dl      = L/Nc
    N       = points.shape[0]
    for p in range(N):
        x,y,z,m = points[p,:]
        i = [math.floor(x/dl-1/2), math.floor(x/dl+1/2)]
        j = [math.floor(y/dl-1/2), math.floor(y/dl+1/2)]
        k = [math.floor(z/dl-1/2), math.floor(z/dl+1/2)]
        dx = [dl*(i[0] + 3/2) - x, x + dl*(1/2 - i[1])]
        dy = [dl*(j[0] + 3/2) - y, y + dl*(1/2 - j[1])]
        dx = [dl*(k[0] + 3/2) - z, z + dl*(1/2 - k[1])]
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    try:
                        density[i[a],j[b],k[c]] += m*dx[a]*dy[b]*dz[c]/(dl**6)
                    except:
                        pass
                        # Cell of the density Field that 
                        # particle would contribute to is out of bounds

def KernelCell(i,j,k,Nc):
    G = 1 
    if i == 0 and j == 0 and k == 0:
        return 0

    if i <= Nc/2:
        kx = 2*np.pi*i/Nc
    else:
        kx = 2*np.pi*(i-Nc)/Nc
    if j <= Nc/2:
        ky = 2*np.pi*j/Nc
    else:
        ky = 2*np.pi*(j-Nc)/Nc
    if k <= Nc/2:
        kz = 2*np.pi*k/Nc
    else:
        kz = 2*np.pi*(k-Nc)/Nc

    return -np.pi*G/(np.sin(kx/2)**2 + np.sin(ky/2)**2 + np.sin(kz/2)**2)

def PotentialField(density, Nc, L):
    fft_density = fft.rfftn(density)
    Nx, Ny, Nz = fft_density.shape
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                fft_density[i,j,k] *= KernelCell(i,j,k,Nc)

    fudge = (L/Nc)**2
    return fft.irfftn(fft_density, density.shape)*fudge

def ForceField(potential, L):
    Nc    = potential.shape[0]
    h     = L/Nc
    force = np.zeros((Nc, Nc, Nc, 3))

    force[0,:,:,0]    = (potential[1,:,:] - potential[0,:,:])/h
    force[Nc-1,:,:,0] = (potential[Nc-1,:,:] - potential[Nc-2,:,:])/h
    for i in range(1,Nc-1):
        force[i,:,:,0] = (potential[i+1,:,:] - potential[i-1,:,:])/(2*h)

    force[:,0,:,1]    = (potential[:,1,:] - potential[:,0,:])/h
    force[:,Nc-1,:,1] = (potential[:,Nc-1,:] - potential[:,Nc-2,:])/h
    for j in range(1,Nc-1):
        force[:,j,:,1] = (potential[:,j+1,:] - potential[:,j-1,:])/(2*h)

    force[:,:,0,2]    = (potential[:,:,1] - potential[:,:,0])/h
    force[:,:,Nc-1,2] = (potential[:,:,Nc-1] - potential[:,:,Nc-2])/h
    for k in range(1,Nc-1):
        force[:,:,k,2] = (potential[:,:,k+1] - potential[:,:,k-1])/(2*h)

    return -force

def ForceCIC(force, points, L):
    N      = points.shape[0]
    Nc     = force.shape[0]
    dl     = L/Nc
    pforce = np.zeros((N,4))
    for p in range(N):
        x,y,z = points[p,:3]
        i  = [math.floor(x/dl-1/2), math.floor(x/dl+1/2)]
        j  = [math.floor(y/dl-1/2), math.floor(y/dl+1/2)]
        k  = [math.floor(z/dl-1/2), math.floor(z/dl+1/2)]
        # print(i,j,k)
        dx = [dl*(i[0] + 3/2) - x, x + dl*(1/2 - i[1])]
        dy = [dl*(j[0] + 3/2) - y, y + dl*(1/2 - j[1])]
        dz = [dl*(k[0] + 3/2) - z, z + dl*(1/2 - k[1])]
        # print(dx,dy,dz)
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    try:
                        pforce[p,:3] += force[i[a],j[b],k[c],:]*dx[a]*dy[b]*dz[c]/(dl**3)
                    except:
                        pass
                        # Cells of the force field that the particle 
                        # interacts with are out of bounds
    return pforce

def ForceComputation(xvec, Nc, L):
    density   = DensityField(xvec, Nc, L)
    potential = PotentialField(density, Nc, L)
    force     = ForceField(potential, L)
    pforce    = ForceCIC(force, xvec, L)
    return pforce

