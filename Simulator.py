import time
import math
import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from scipy.stats import cumfreq

def InitialConditions(R, N, c, m, seed):
    '''Returns a list of points corresponding to a spherical mass distribution

    Keyword arguments:
    R    -- the radius of the sphere particles are distributed inside
    N    -- the number of particles to generate
    c    -- the center of the sphere
    m    -- the mass given to each particle
    seed -- the seed to use for random number generation
    '''
    rng = default_rng(seed)
    x1  = rng.random(N)
    x2  = rng.random(N)
    x3  = rng.random(N)

    r   = R*x1**(1/3)
    phi = 2*np.pi*x2
    the = np.arccos(1-2*x3)

    coords      = np.empty((N,4))
    velocities  = np.zeros((N,4))
    coords[:,0] = r*np.sin(the)*np.cos(phi) + c[0]
    coords[:,1] = r*np.sin(the)*np.sin(phi) + c[1]
    coords[:,2] = r*np.cos(the) + c[2]
    coords[:,3] = m

    return (coords, velocities)

def TimeStep(v, a, dl):
    '''Computes the appropriate time step value for one iteration of Verlet's

    Keyword arguements:
    v  -- a numpy array corresponding to the velocities of each point
    a  -- a numpy array corresponding to the accelerations of each point
    dl -- the length of a simulation grid cell
    '''
    lmd   = .5
    v_max = max([sum(v[i,:]**2)**(1/2) for i in range(v.shape[0])])
    if v_max == 0:
        v_max = .01
    a_max = max([sum(a[i,:]**2)**(1/2) for i in range(v.shape[0])])
    return lmd*min(dl/v_max, (dl/a_max)**(1/2))

def verlet(t0, x0, v0, tf, Nc, L):
    '''An implementation of Verlet Integration

    Keyword arguments:
    t0 -- starting time
    x0 -- initial numpy array of particle coordinates
    v0 -- initial numpy array of particle velocities
    tf -- stopping time
    Nc -- the number of grid cells in one dimension of the cubic grid
    L  -- the length of one side of the simulation space
    '''
    # Init
    t  = t0
    x  = x0
    v  = v0
    dl = L/Nc
    a  = ForceComputation(x, Nc, L)
    h  = TimeStep(v, a, dl)
    vhalf = v + h*a/2
    # Loop over time steps
    while t < tf:
        x_old = x
        v_old = v
        x = x_old + h*vhalf
        a = ForceComputation(x, Nc, L)
        k = h*a
        v = vhalf+k/2
        t = t + h
        vhalf = vhalf + k
        h = TimeStep(v, a, dl)

    return (t, x, v, a)

def VerletHeader(x0, v0, ts, Nc, L):
    '''Computes Verlet's method with regular snapshots of simulation progress

    Keyword arguments:
    x0 -- initial numpy array of particle coordinates
    v0 -- initial numpy array of particle velocities
    ts -- an array of times to get snapshots of
    Nc -- the number of grid cells in one dimension of the cubic grid
    L  -- the length of one side of the simulation space
    '''
    t0 = ts[0]
    x  = [x0]
    v  = [v0]
    a  = [ForceComputation(x0, Nc, L)]
    for t in ts[1:]:
        t0, xv, vv, av = verlet(t0, x[-1], v[-1], t, Nc, L)
        x.append(xv)
        v.append(vv)
        a.append(av)

    return (x, v, a)

if __name__=="__main__":
    '''
    The proceeding block runs the simulation for a grid with 
    side lengths of one unit and 128 cells. The starting 
    mass distribution consists of a sphere with a radius of 
    1/4 a unit centered at the center of the grid with 32**3 
    points. We run the simulation for tdyn time where tdyn 
    is the dynamical time of the system, and we take snapshots
    of the system in intervals of 1/20 of the total time.

    Finally, we graph the state of the system from two viewing 
    planes, the mass profile, and the acceleration profile at 
    each snapshot.
    '''
    L     = 1
    Nc    = 128
    R     = L/4
    c     = [L/2, L/2, L/2]
    N     = 32**3
    m     = .1 # kg
    seed  = 13
    tdyn  = ( (np.pi**2*R**3)/(4*G*m*N) )**(1/2)
    t     = tdyn*np.array([i*.05 for i in range(21)])
    start_t = time.time()
    coord, velocities = InitialConditions(R, N, c, m, seed)
    x, v, a = VerletHeader(coord, velocities, t, Nc, L)
    print(time.time() - start_t)
    print("done!")
    
    color    = [i/N for i in range(N)]
    center   = np.array(c)
    r_sample = np.linspace(0, R, 100)
    Init_M   = m*N*r_sample**3/R**3

    xsize = 36
    ysize = 190
    plt.figure(figsize=(xsize,ysize))
    plt.subplots_adjust(wspace=.38, hspace=.35)

    for i in range(21):
        plt.subplot(21, 4, 4*i+1)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title(f"X-Y t={t[i]}")
        plt.scatter(x[i][:,0], x[i][:,1], c=color)
    
    
        plt.subplot(21, 4, 4*i+2)
        plt.xlabel('$z$')
        plt.ylabel('$y$')
        plt.title(f"Z-Y t={t[i]}")
        plt.scatter(x[i][:,2], x[i][:,1], c=color)
    
    
        plt.subplot(21, 4, 4*i+3)
        plt.xlabel('$r$')
        plt.ylabel('$M(<r)$')
        plt.title(f"Mass Profile t={t[i]}")
        r_calc = np.zeros(N)
        for j in range(N):
            r_calc[j] = sum((center - x[i][j,:3])**2)**(1/2)
        mass_prof = cumfreq(r_calc, numbins=100, weights=[.1]*N)
        mass_dom  = mass_prof.lowerlimit + np.linspace(0, mass_prof.binsize*mass_prof.cumcount.size, mass_prof.cumcount.size)
        plt.bar(mass_dom, mass_prof.cumcount, width=mass_prof.binsize, label='Computed Mass Profile')
        plt.plot(r_sample, Init_M, "C1-", label='Analytical Mass Profile')
        plt.legend()


        plt.subplot(21, 4, 4*i+4)
        plt.title(f"Acceleration Profile t={t[i]}")
        plt.xlabel('$r$')
        plt.ylabel('$a$')
        a_mag = np.zeros(N)
        for j in range(N):
            a_mag[j] = sum(a[i][j,:3]**2)**(1/2)
        a_pro = mass_prof.cumcount/mass_dom**2
        plt.scatter(r_calc, a_mag, c=color, label='Computed Acceleration')
        plt.plot(mass_dom, a_pro, "C1o", label='Analytical Acceleration Profile')
        plt.legend()

    plt.show()
