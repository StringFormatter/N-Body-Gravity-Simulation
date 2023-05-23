import math
import numpy as np
from numpy import fft
import matplotlib        as mpl
import matplotlib.pyplot as plt

def DensityField(points, Nc, L):
    '''Broadcasts the mass of a list of points into a density grid 

    Uses a Cloud in Cell (CIC) scheme where each point is treated
    as the size of one grid cell and each actual grid cell is assigned 
    mass proportional to the amount of overlap with the point cell.

    Keyword arguments:
    points -- a numpy array containing the coordinates and mass of each point
    Nc     -- the number of grid cells in one dimension of the cubic grid
    L      -- the length of one side of the simulation space
    '''
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
        dz = [dl*(k[0] + 3/2) - z, z + dl*(1/2 - k[1])]
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    try:
                        density[i[a],j[b],k[c]] += m*dx[a]*dy[b]*dz[c]/(dl**6)
                    except:
                        pass
                        # Cell of the density Field that 
                        # particle would contribute to is out of bounds
    return density

def KernelCell(i,j,k,Nc):
    '''Computes the value of our kernel in Fourier space

    The traditional kernel for Gravitational Poisson's equation is 
    isotropic, but we use an anisotropic form which solves the finite-difference
    approximation of Poisson's equation.

    Keyword arguments:
    i  -- integer wavenumber for the "x" component
    j  -- integer wavenumber for the "y" component
    k  -- integer wavenumber for the "z" component
    Nc -- the number of wavenumbers along one dimension
    '''
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
    '''Solves' Poisson's equation to find gravitational potential

    The differential form of Gauss' law for gravity can be transformed 
    into Poisson's equation via relating gravitational potential to 
    density. We can solve Poisson's equation with a Green's function, 
    which corresponds to a convolution of the density field with a 
    specific kernel. Therefore, we find gravitational potential by 
    taking the product of a Real Fourier transform of the density field 
    and the Kernel, also in Fourier space.

    Keyword arguments:
    density -- a 3d numpy array containing the mass density of each grid cell
    Nc      -- the number of grid cells in one direction of the cubic grid
    L       -- the length of one side of the simulation space
    '''
    fft_density = fft.rfftn(density)
    Nx, Ny, Nz = fft_density.shape
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                fft_density[i,j,k] *= KernelCell(i,j,k,Nc)

    fudge = (L/Nc)**2
    return fft.irfftn(fft_density, density.shape)*fudge

def ForceField(potential, L):
    '''Differentiates gravitational potentials to get forces

    Keyword arguments:
    potential -- a 3d numpy array containing the gravitational potentials
    L         -- the length of one side of the simulation space
    '''
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
    '''Interpolates the force field to find the force experienced by each point

    Again, a Cloud in Cell scheme is used to interpolate, where the particle is
    treated as a grid cell, and the amount of force a given cell contributes is
    proportional to the amount of overlap between the particle cell and the 
    given cell.

    Keyword arguments:
    force  -- a 3d numpy array containing the force field
    points -- a numpy array containing the coordinates and mass of each point
    L      -- the length of one side of the simulation space
    '''
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

def ForceComputation(points, Nc, L):
    '''Uses the particle mesh scheme to find the force acting on each particle

    Keyword arguments:
    points -- a numpy array containing the coordinates and mass of each point
    Nc     -- the number of grid cells in one dimension of the cubic grid
    L      -- the length of one side of the simulation space
    '''
    density   = DensityField(points, Nc, L)
    potential = PotentialField(density, Nc, L)
    force     = ForceField(potential, L)
    pforce    = ForceCIC(force, points, L)
    return pforce
