import numpy as np
from mpi4py import MPI
from collections import namedtuple
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from eigentools import Eigenproblem

import time
# import matplotlib.pyplot as plt
comm = MPI.COMM_WORLD

def evp_name(params):
    """Generate EVP run name."""
    name = "eigenmodes-" + "-".join([f"{key}-{getattr(params, key)}" for key in evp_params._fields])
    name = name.replace(".", "_")
    return name

evp_params = namedtuple('evp_params', ['Nx', 'Re', 'Rm', 'R', 'q', 'kz', 'ky'])

def build_evp(params, save_dir, **kw):
    # fixed params
    Lx = np.pi
    B = 1

    Ly = 2 * np.pi / params.ky
    Lz = 2 * np.pi / params.kz

    # Parameters
    Re = params.Re
    Rm = params.Rm
    if Re != 0 and Rm != 0:
        diffusion = True
    else:
        diffusion = False
    Nx = params.Nx
    R = params.R
    q = params.q
    kx = np.pi/Lx
    S = -R*B*kx*np.sqrt(q)
    f_param =  R*B*kx/np.sqrt(q)

    # Build Fourier basis for x, y with prescribed kx, ky as the fundamental modes
    Ny = 4
    Nz = 4

    # Bases
    coords = d3.CartesianCoordinates('y', 'z', 'x')
    # dist = d3.Distributor(coords, dtype=np.complex128, comm=MPI.COMM_SELF)
    dist = d3.Distributor(coords, dtype=np.complex128)
    xbasis = d3.ChebyshevT(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2))
    ybasis = d3.ComplexFourier(coords['y'], size=Ny, bounds=(0, Ly))
    zbasis = d3.ComplexFourier(coords['z'], size=Nz, bounds=(0, Lz))

    y, z, x = dist.local_grids(ybasis, zbasis, xbasis)

    # Fields
    bases = (ybasis, zbasis, xbasis)
    sigma = dist.Field(name='sigma')
    p = dist.Field(name='p', bases=bases)
    b = dist.VectorField(coords, name='b', bases=bases)
    u = dist.VectorField(coords, name='u', bases=bases)
    tau_c1 = dist.Field(name='tau_c1')
    tau_c2 = dist.Field(name='tau_c2')

    boundary_bases = (ybasis, zbasis)
    if diffusion:
        tau_b1 = dist.VectorField(coords, name='tau_b1', bases=boundary_bases)
        tau_u1 = dist.VectorField(coords, name='tau_u1', bases=boundary_bases)
        tau_b2 = dist.VectorField(coords, name='tau_b2', bases=boundary_bases)
        tau_u2 = dist.VectorField(coords, name='tau_u2', bases=boundary_bases)
    else:
        tau_b1 = dist.Field(name='tau_b1', bases=boundary_bases)
        tau_u1 = dist.Field(name='tau_u1', bases=boundary_bases)
        tau_b2 = dist.Field(name='tau_b2', bases=boundary_bases)
        tau_u2 = dist.Field(name='tau_u2', bases=boundary_bases)

    # 2 omega
    f = dist.VectorField(coords, name = 'f',bases=(xbasis,))

    # background velocity field
    u0 = dist.VectorField(coords, name='u0', bases=(xbasis,))
    b0 = dist.VectorField(coords, name='b0',bases=(xbasis,))
    #b0 = dist.VectorField(coords, name='b0')

    # Substitutions
    ey, ez, ex = coords.unit_vector_fields(dist)
    lift_basis = xbasis.derivative_basis(2)
    lift = lambda A,i: d3.Lift(A, lift_basis, i)
    lift_basis1 = xbasis.derivative_basis(1)
    lift1 = lambda A,i: d3.Lift(A, lift_basis1, i)

    dt = lambda A: sigma*A

    f['g'][1] = f_param
    u0['g'][0] = S*x
    b0['g'][1] = B
    j = d3.curl(b)
    # Problem
    # First-order form: "div(f)" becomes "trace(grad_f)"
    # First-order form: "lap(f)" becomes "div(grad_f)"
    tau_u = lift(tau_u1,-1) + lift(tau_u2, -2)
    tau_b = lift(tau_b1,-1) + lift(tau_b2, -2)
    tau_div = lift1(tau_c1, -1) + tau_c2
    if diffusion:
        problem = d3.EVP([p, u, b, tau_c1, tau_c2, tau_u1, tau_b1, tau_u2, tau_b2], namespace=locals(), eigenvalue=sigma)
    else:
        problem = d3.EVP([p, u, b, tau_c1, tau_c2, tau_u1, tau_b1], namespace=locals(), eigenvalue=sigma)
    if diffusion:
        problem.add_equation("dt(b) - lap(b)/Rm + tau_b + u0@grad(b) - b0@grad(u) - b@grad(u0)= 0")
        problem.add_equation("dt(u) - lap(u)/Re + tau_u + dot(u0,grad(u)) + dot(u,grad(u0)) + grad(p) - b0@grad(b) + cross(f, u) = 0" )
        problem.add_equation("div(u) + tau_div = 0")
        problem.add_equation("integ(ex@tau_u2) = 0")
        problem.add_equation("integ(p) = 0") # Pressure gauge
        # problem.add_equation("u(x=-Lx/2) = 0")
        # problem.add_equation("u(x=Lx/2) = 0")
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
    else:
        problem.add_equation("dt(b) + ex*lift1(tau_b1,-1)+ ex*lift1(tau_b2,-2) - curl(cross(u0,b)) - curl(cross(u,b0))= 0")
        problem.add_equation("dt(u) + ex*lift1(tau_u1,-1)+ ex*lift1(tau_u2,-2) + dot(u0,grad(u)) + dot(u,grad(u0)) + grad(p) - curl(Co*cross(b,b0)) + 2*cross(omega, u) = 0" )
        problem.add_equation("div(u) + lift1(tau_c1,-1) + tau_c2 = 0")
        problem.add_equation("integ(p) = 0") # Pressure gauge
        problem.add_equation("ex@u(x=-Lx/2) = 0")
        problem.add_equation("ex@u(x=Lx/2) = 0")
        problem.add_equation("ex@b(x=-Lx/2) = 0")
        problem.add_equation("ex@b(x=Lx/2) = 0")

    return problem

def dense_evp(params, save_dir="data", reload=False, subproblem=(0,1,None), **kw):
    """Solve EVP for all eigenmodes."""

    factor = 3/2
    Nx_hi = int(factor*params.Nx)
    # Get EVP name
    name = evp_name(params)
    logger.info(f"Computing EVP: {name}")
    logger.info(f"Nx_low = {params.Nx}, Nx_high = {Nx_hi}")

    params_hires = evp_params(
        Nx = Nx_hi,
        Re = params.Re,
        Rm = params.Rm,
        R = params.R,
        q =  params.q,
        kz = params.kz,
        ky = params.ky,
    )
    ncc_cutoff=0
    evp = build_evp(params, save_dir=save_dir)
    evp_hires = build_evp(params_hires, save_dir=save_dir)
    EVP = Eigenproblem(evp, EVP_secondary=evp_hires, ncc_cutoff=ncc_cutoff, reject='distance')
    if reload:
        load_filename = f"{save_dir}/{name}_etools"
        with np.load(load_filename+'.npz', allow_pickle=True) as data:
            EVP.evalues_primary = data['all_eigenvalues']
            EVP.evalues_secondary = data['all_eigenvalues_hires']
            EVP.solver.eigenvalues = EVP.evalues_primary
            EVP.solver.right_eigenvectors = data['all_eigenvectors']
            EVP.solver.modified_left_eigenvectors = data['all_modified_left_eigenvectors']
        sp =  EVP.solver.subproblems_by_group[subproblem]
        EVP.solver.eigenvalue_subproblem = sp
        EVP.reject_spurious()
        logger.info(f"{len(EVP.evalues)} good eigenmodes")
        # calc_c = lambda ev,kz: 1j*ev/kz
        # c = calc_c(EVP.evalues, params.kz)
        # c_pri = calc_c(EVP.evalues_primary, params.kz)
        # c_sec = calc_c(EVP.evalues_secondary, params.kz)
        # logger.info(f"c.imag max = {np.max(c.imag)}")
        # logger.info(f"c_pri.imag max = {np.nanmax(c_pri.imag)}")
        # logger.info(f"c_sec.imag max = {np.nanmax(c_sec[c_sec.imag < 0].imag)}")
    else:
        t0 = time.time()
        #solver.solve_dense(sp)
        #EVP.solve(subproblem=subproblem)
        sp =  EVP.solver.subproblems_by_group[subproblem]
        EVP.solve(subproblem=subproblem, left=True)
        t1 = time.time()
        logger.info(f"EVP solve time: {t1-t0:.2f}")
        logger.info(f"Matrix size: {sp.L_min.shape}")
        # Save eigenmodes
    np.savez(
        f"{save_dir}/{name}_etools",
        eigenvalues=EVP.evalues,
        all_eigenvalues=EVP.evalues_primary,
        all_eigenvalues_hires=EVP.evalues_secondary,
        all_eigenvectors=EVP.solver.right_eigenvectors,
        all_modified_left_eigenvectors=EVP.solver.modified_left_eigenvectors,
        **params._asdict())


    return EVP

if __name__ == "__main__":
    # Parameters
    R = 1.001
    ky = 0.263
    kz = 0.447
    Re = 100000
    subproblem = (1,1,None)
    params = evp_params(
        Re = Re,
        Rm = Re,
        Nx = 128,
        R = R,
        q = 3/4,
        ky = ky,
        kz = kz,
    )
    EVP = dense_evp(params,subproblem=subproblem)
    growth = (np.max(EVP.evalues.real))
    print(f"max growth rate = {growth}")
    EVP.solver.print_subproblem_ranks()
# np.savez(
#     f"{save_dir}/{name}_etools",
#     eigenvalues=solver.eigenvalues,
#     eigenvectors=solver.eigenvectors,
#     )
