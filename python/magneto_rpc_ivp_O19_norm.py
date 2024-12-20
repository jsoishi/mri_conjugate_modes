import numpy as np
from mpi4py import MPI
from collections import namedtuple
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

import time
import matplotlib.pyplot as plt
comm = MPI.COMM_WORLD

def ivp_name(params):
    """Generate IVP run name."""
    name = "IVP-" + "-".join([f"{key}-{getattr(params, key)}" for key in ivp_params._fields])
    name = name.replace(".", "_")
    return name

ivp_params = namedtuple('ivp_params', ['Nx', 'Ny', 'Nz', 'Re', 'Rm', 'R', 'q', 'kz', 'ky', 'no_slip'])

dealias = 3/2
timestepper = d3.RK222
max_timestep = 0.0675

# Parameters
params = ivp_params(
    Nx = 128,
    Ny = 64,
    Nz = 64,
    q = 0.75,
    R = 1.001,
    ky = 0.263,
    kz = 0.447,
    Re = 100,
    Rm = 100,
    no_slip = False
)
name = ivp_name(params)
# fixed params
Lx = np.pi
B = 1
q = 4/3

Nx = params.Nx
Ny = params.Ny
Nz = params.Nz
q  = params.q
R  = params.R
ky = params.ky
kz = params.kz
Re = params.Re
Rm = params.Rm
no_slip = params.no_slip
Lx = np.pi
Ly = 2 * np.pi / ky
Lz = 2 * np.pi / kz

# Derived Parameters
S = -R*B*np.sqrt(q)
f_param =  R*B/np.sqrt(q)

# Bases
coords = d3.CartesianCoordinates('y', 'z', 'x')
dist = d3.Distributor(coords, dtype=np.float64, mesh=(8,16))
xbasis = d3.ChebyshevT(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2),dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
y, z, x = dist.local_grids(ybasis, zbasis, xbasis)

# Fields
bases = (ybasis, zbasis, xbasis)
sigma = dist.Field(name='sigma')
p = dist.Field(name='p', bases=bases)
phi = dist.Field(name='phi', bases=bases)
b = dist.VectorField(coords, name='b', bases=bases)
u = dist.VectorField(coords, name='u', bases=bases)
tau_c1 = dist.Field(name='tau_c1')
tau_c2 = dist.Field(name='tau_c2')
tau_bdiv1 = dist.Field(name='tau_bdiv1')
tau_bdiv2 = dist.Field(name='tau_bdiv2')

boundary_bases = (ybasis, zbasis)
tau_b1 = dist.VectorField(coords, name='tau_b1', bases=boundary_bases)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=boundary_bases)
tau_b2 = dist.VectorField(coords, name='tau_b2', bases=boundary_bases)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=boundary_bases)

# backgrounds
f = dist.VectorField(coords, name = 'f',bases=(xbasis,))
u0 = dist.VectorField(coords, name='u0', bases=(xbasis,))
b0 = dist.VectorField(coords, name='b0',bases=(xbasis,))

# Substitutions
ey, ez, ex = coords.unit_vector_fields(dist)
lift_basis = xbasis.derivative_basis(2)
lift = lambda A,i: d3.Lift(A, lift_basis, i)
lift_basis1 = xbasis.derivative_basis(1)
lift1 = lambda A,i: d3.Lift(A, lift_basis1, i)

f['g'][1] = f_param
u0['g'][0] = S*x
b0['g'][1] = B
j = d3.curl(b)
tau_u = lift(tau_u1,-1) + lift(tau_u2, -2)
tau_b = lift(tau_b1,-1) + lift(tau_b2, -2)
tau_div = lift1(tau_c1, -1) + tau_c2
tau_bdiv = lift1(tau_bdiv1, -1) + tau_bdiv2

problem = d3.IVP([p, u, b, phi, tau_c1, tau_c2, tau_bdiv1, tau_bdiv2, tau_u1, tau_b1, tau_u2, tau_b2], namespace=locals())

problem.add_equation("dt(b) - lap(b)/Rm + tau_b + u0@grad(b) - b0@grad(u) - b@grad(u0) + grad(phi)= curl(cross(u,b))")
problem.add_equation("dt(u) - lap(u)/Re + tau_u + dot(u0,grad(u)) + dot(u,grad(u0)) + grad(p) - b0@grad(b) + cross(f, u) = -u@grad(u)") 
problem.add_equation("div(u) + tau_div = 0")
problem.add_equation("div(b) + tau_bdiv = 0")

problem.add_equation("integ(ex@tau_u2) = 0")
problem.add_equation("integ(ex@tau_b2) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge
problem.add_equation("integ(phi) = 0") # magnetic "pressure" gauge

if no_slip:
    logger.info("Running with no slip boundary conditions")
    problem.add_equation("u(x=-Lx/2) = 0")
    problem.add_equation("u(x=Lx/2) = 0")
else:
    logger.info("Running with free slip boundary conditions")
    problem.add_equation("ex@u(x=-Lx/2) = 0")
    problem.add_equation("ex@u(x=Lx/2) = 0")
    problem.add_equation("ex@(grad(ey@u)(x=-Lx/2)) = 0")
    problem.add_equation("ex@(grad(ey@u)(x=Lx/2)) = 0")
    problem.add_equation("ex@(grad(ez@u)(x=-Lx/2)) = 0")
    problem.add_equation("ex@(grad(ez@u)(x=Lx/2)) = 0")

problem.add_equation("ex@b(x=-Lx/2) = 0")
problem.add_equation("ex@b(x=Lx/2) = 0")
problem.add_equation("ey@j(x=-Lx/2) = 0")
problem.add_equation("ey@j(x=Lx/2) = 0")
problem.add_equation("ez@j(x=-Lx/2) = 0")
problem.add_equation("ez@j(x=Lx/2) = 0")

solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = 150

vol = Lx*Ly*Lz
integ = lambda A: d3.Integrate(d3.Integrate(d3.Integrate(A, 'y'), 'z'), 'x')
volavg = lambda A: integ(A)/vol
yz_avg = lambda A: d3.Integrate(d3.Integrate(A, 'y'), 'z')/(Ly*Lz)
KE = 0.5*u@u
ME = 0.5*b@b

dir_path = f"data/{name}"

snapshots_path = dir_path+"/snapshots"
snapshots = solver.evaluator.add_file_handler(snapshots_path, sim_dt=1, max_writes=1)
snapshots.add_tasks(solver.state)
scalars_path = dir_path+"/scalars"
scalars = solver.evaluator.add_file_handler(scalars_path, iter=10, max_writes=None)
scalars.add_task(volavg(KE), name='KE')
scalars.add_task(volavg(ME), name='ME')
scalars.add_task(np.sqrt(volavg(d3.dot(u,u))), name='RMS_velocity') 
scalars.add_task(np.sqrt(volavg(d3.div(u)**2)), name='|div_u|')
scalars.add_task(np.sqrt(volavg(d3.div(b)**2)), name='|div_b|')
#scalars.add_task(np.sqrt(volavg(tau_d**2)), name='|tau_d|')
scalars.add_task(np.sqrt(volavg(tau_u@tau_u)), name='|tau_u|')
scalars.add_task(np.sqrt(volavg(tau_b@tau_b)), name='|tau_b|')
scalars.add_task(np.sqrt(volavg(d3.grad(phi)@d3.grad(phi))), name='|grad_phi|')

CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)
CFL.add_velocity(b)

# Initial conditions
ic_potential = dist.VectorField(coords, name='Phi', bases=bases)
ic_potential.fill_random('g', seed=42, distribution='normal', scale=1e-2) # Random noise
ic_potential['g'] *= x * (Lx/2 - x) # Damp noise at walls

A = 1e-5
u_ic = d3.Curl(ic_potential).evaluate()
u_ic.change_scales(1)
u['g'] = A*u_ic['g']

# Main loop
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
    solver.evaluate_handlers()
    solver.log_stats()


