"""
Dedalus script for calculating stability of rotating plane Couette (rpC) flow.

For incompressible hydro with two boundaries, we need two tau terms for each the
velocity and buoyancy. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 rayleigh_benard_evp.py
"""

import numpy as np
from mpi4py import MPI
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

import time
import matplotlib.pyplot as plt
comm = MPI.COMM_WORLD

# Parameters
Co = 0.08
Rm = 16.3
Pm = 1
Reynolds = Rm/Pm 
omega0 = 1
b0_scalar = 1
q = 3/2


Nx = 128
Ny = 64
Nz = 64
# kz = 5
# ky = 5
Lx = 1
# Parameters
# Ly = 2 * np.pi / ky
# Lz = 2 * np.pi / kz
Ly = 10
Lz = 10


dealias = 3/2
stop_sim_time = 30
timestepper = d3.RK222
max_timestep = 0.125/2
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('z', 'y', 'x')
# dist = d3.Distributor(coords, dtype=np.complex128, comm=MPI.COMM_SELF)
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.ChebyshevT(coords['x'], size=Nx, bounds=(-Lx, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

z, y, x = dist.local_grids(zbasis, ybasis, xbasis)
    

# Fields
bases = (zbasis, ybasis, xbasis)
p = dist.Field(name='p', bases=bases)
phi = dist.Field(name='phi', bases=bases)
a = dist.VectorField(coords, name='a', bases=bases)
u = dist.VectorField(coords, name='u', bases=bases)
tau_p = dist.Field(name='tau_p')
tau_phi = dist.Field(name='tau_phi')
tau_a1 = dist.VectorField(coords, name='tau_a1', bases=(zbasis, ybasis))
tau_a2 = dist.VectorField(coords, name='tau_a2', bases=(zbasis, ybasis))
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=(zbasis, ybasis))
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=(zbasis, ybasis))

# inverse Rossby number
omega = dist.VectorField(coords, name = 'omega',bases=(xbasis,))

# background velocity field
u0 = dist.VectorField(coords, name='u0', bases=(xbasis,))
b0 = dist.VectorField(coords, name='b0',bases=(xbasis,))


# Substitutions
ez, ey, ex = coords.unit_vector_fields(dist)
lift_basis = xbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ex*lift(tau_u1) # First-order reduction
grad_a = d3.grad(a) + ex*lift(tau_a1) # First-order reduction

b = d3.curl(a)
j = d3.curl(b)
omega['g'][0] = 1
u0['g'][1] = -q*omega0*x
b0['g'][0] = b0_scalar
# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, u, a, phi, tau_p, tau_u1, tau_u2, tau_a1, tau_a2, tau_phi], namespace=locals())
# problem.add_equation("dt(a) + lift(tau_a2) + grad(phi) - div(grad_a)/Rm - ((grad(a)@u0-u0@grad(a)) + cross(u,b0))= cross(u0,b0) + cross(u,b)")
# problem.add_equation("dt(u) + dot(u0,grad(u)) + dot(u,grad(u0)) + grad(p) + lap(Co*(cross(a, b0)))  \
#                     - div(grad_u)/Reynolds - 2*cross(omega, u) + lift(tau_u2) = -dot(u0,grad(u0)) - dot(u,grad(u))+2*cross(omega, u0)+Co*(cross(j, b))")
# problem.add_equation("dt(a) + lift(tau_a2) + grad(phi) - div(grad_a)/Rm - ((grad(a)@u0-u0@grad(a)) + cross(u,b0))=0")
# problem.add_equation("dt(u) + dot(u0,grad(u)) + dot(u,grad(u0)) + grad(p) + lap(Co*(cross(a, b0)))  \
                    # - div(grad_u)/Reynolds - 2*cross(omega, u) + lift(tau_u2) = 0")
problem.add_equation("dt(a) + lift(tau_a2) + grad(phi) - div(grad_a)/Rm - ((grad(a)@u0-u0@grad(a)) + cross(u,b0))= 0")
problem.add_equation("dt(u) + dot(u0,grad(u)) + dot(u,grad(u0)) + grad(p) + lap(Co*(cross(a, b0)))  \
                    - div(grad_u)/Reynolds - 2*cross(omega, u) + lift(tau_u2) = 0")
problem.add_equation("trace(grad_a) + tau_phi = 0")
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge
problem.add_equation("integ(phi) = 0")
problem.add_equation("u(x=-Lx) = 0")
problem.add_equation("u(x=Lx) = 0")
problem.add_equation("ey@a(x=-Lx) = 0")
problem.add_equation("ey@a(x=Lx) = 0")
problem.add_equation("ez@a(x=-Lx) = 0")
problem.add_equation("ez@a(x=Lx) = 0")
problem.add_equation("phi(x=-Lx) = 0")
problem.add_equation("phi(x=Lx) = 0")
# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time


vol = Lx*Ly*Lz
integ = lambda A: d3.Integrate(d3.Integrate(d3.Integrate(A, 'y'), 'z'), 'x')
avg = lambda A: integ(A)/vol
yz_avg = lambda A: d3.Integrate(d3.Integrate(A, 'y'), 'z')/(Ly*Lz)
KE = 0.5*u@u
ME = 0.5*b@b

# Analysis
snapshots = solver.evaluator.add_file_handler('mri_ivp/linear/snapshots', sim_dt=1, max_writes=50)
snapshots.add_task(u, name='velocity field') 
snapshots.add_task(u(y=Ly/2), name='u mid y')
snapshots.add_task(u(z=Lz/2), name='u mid z')

traces = solver.evaluator.add_file_handler('mri_ivp/linear/traces', sim_dt=0.5, max_writes=None)
traces.add_task(avg(KE), name='KE')
traces.add_task(avg(ME), name='ME')
traces.add_task(np.sqrt(avg(d3.dot(u,u))), name='RMS_velocity') 
# CFL
# CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
#              max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)

CFL.add_velocity(u)
CFL.add_velocity(a)
# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)

# Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            # max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
