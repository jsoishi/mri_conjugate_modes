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


def max_growth_rate(Reynolds, Rossby, ky, kz, Nx, NEV=10, target=0):
    """Compute maximum linear growth rate."""

    # Parameters
    Lx = 1
    # Build Fourier basis for x, y with prescribed kx, ky as the fundamental modes
    Ny = 2
    Nz = 2

    Ly = 2 * np.pi / ky
    Lz = 2 * np.pi / kz
    # Bases
    coords = d3.CartesianCoordinates('z', 'y', 'x')
    dist = d3.Distributor(coords, dtype=np.complex128, comm=MPI.COMM_SELF)
    xbasis = d3.ChebyshevT(coords['x'], size=Nx, bounds=(-Lx, Lx))
    ybasis = d3.ComplexFourier(coords['y'], size=Ny, bounds=(0, Ly))
    zbasis = d3.ComplexFourier(coords['z'], size=Nz, bounds=(0, Lz))

    z, y, x = dist.local_grids(zbasis, ybasis, xbasis)
        

    # Fields
    bases = (zbasis, ybasis, xbasis)
    sigma = dist.Field(name='sigma')
    p = dist.Field(name='p', bases=bases)
    u = dist.VectorField(coords, name='u', bases=bases)
    tau_p = dist.Field(name='tau_p')
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=(zbasis, ybasis))
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=(zbasis, ybasis))

    # inverse Rossby number
    inv_Ro = dist.VectorField(coords, name = '1/Ro')
    # background velocity field
    u0 = dist.VectorField(coords, name='u', bases=bases)

    # Substitutions
    ez, ey, ex = coords.unit_vector_fields(dist)
    lift_basis = xbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    grad_u = d3.grad(u) + ex*lift(tau_u1) # First-order reduction
    dt = lambda A: sigma*A

    inv_Ro['g'][2] = 1/Rossby
    u0['g'][1] = -x
    # Problem
    # First-order form: "div(f)" becomes "trace(grad_f)"
    # First-order form: "lap(f)" becomes "div(grad_f)"
    problem = d3.EVP([p, u, tau_p, tau_u1, tau_u2], namespace=locals(), eigenvalue=sigma)
    problem.add_equation("trace(grad_u) + tau_p = 0")
    problem.add_equation("dt(u) + dot(u0,grad(u)) + dot(u,grad(u0)) - div(grad_u)/Reynolds + grad(p) - cross(inv_Ro, u) + lift(tau_u2) = 0")
    problem.add_equation("u(x=-Lx) = 0")
    problem.add_equation("u(x=Lx) = 0")
    problem.add_equation("integ(p) = 0") # Pressure gauge

    # Solver
    solver = problem.build_solver(entry_cutoff=0)
    growth = []
    for p in solver.subproblems:
        solver.solve_sparse(p, NEV, target=target)
        growth.append(np.max(solver.eigenvalues.real))
    return np.max(growth)


if __name__ == "__main__":

    import time
    import matplotlib.pyplot as plt
    comm = MPI.COMM_WORLD

    # Parameters
    Nx = 64
    Reynolds = 1710
    Rossby = 100
    kz_global = np.linspace(1.0, 3.25, 50)
    ky = 3.0
    NEV = 10

    # Compute growth rate over local wavenumbers
    kz_local = kz_global[comm.rank::comm.size]
    t1 = time.time()
    growth_local = np.array([max_growth_rate(Reynolds, Rossby, ky, kz, Nx, NEV=NEV) for kz in kz_local])
    t2 = time.time()
    logger.info('Elapsed solve time: %f' %(t2-t1))

    # Reduce growth rates to root process
    growth_global = np.zeros_like(kz_global)
    growth_global[comm.rank::comm.size] = growth_local
    if comm.rank == 0:
        comm.Reduce(MPI.IN_PLACE, growth_global, op=MPI.SUM, root=0)
    else:
        comm.Reduce(growth_global, growth_global, op=MPI.SUM, root=0)

    # Plot growth rates from root process
    if comm.rank == 0:
        plt.figure(figsize=(6,4))
        plt.plot(kz_global, growth_global, '.')
        plt.xlabel(r'$k_z$')
        plt.ylabel(r'$\mathrm{Re}(\sigma)$')
        plt.title(r'rpC growth rates ($\mathrm{Re} = %.2f, \; \mathrm{Ro} = %.2f$)' %(Reynolds, Rossby))
        plt.tight_layout()
        plt.savefig('growth_rates.pdf')
