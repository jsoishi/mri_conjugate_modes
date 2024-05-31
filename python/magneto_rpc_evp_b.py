import numpy as np
from mpi4py import MPI
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

def A_max_growth_rate(Reynolds, omega0, Rm, Co, b0_scalar, q, ky, kz, Nx, NEV=10, target=5):
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
    # dist = d3.Distributor(coords, dtype=np.complex128, comm=MPI.COMM_SELF)
    dist = d3.Distributor(coords, dtype=np.complex128)
    xbasis = d3.ChebyshevT(coords['x'], size=Nx, bounds=(-Lx, Lx))
    ybasis = d3.ComplexFourier(coords['y'], size=Ny, bounds=(0, Ly))
    zbasis = d3.ComplexFourier(coords['z'], size=Nz, bounds=(0, Lz))

    z, y, x = dist.local_grids(zbasis, ybasis, xbasis)
        

    # Fields
    bases = (zbasis, ybasis, xbasis)
    sigma = dist.Field(name='sigma')
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
    dt = lambda A: sigma*A

    b = d3.curl(a)
    omega['g'][0] = 1
    u0['g'][1] = -q*omega0*x
    b0['g'][0] = b0_scalar
    # Problem
    # First-order form: "div(f)" becomes "trace(grad_f)"
    # First-order form: "lap(f)" becomes "div(grad_f)"
    problem = d3.EVP([p, u, a, phi, tau_p, tau_u1, tau_u2, tau_a1, tau_a2, tau_phi], namespace=locals(), eigenvalue=sigma)
    problem.add_equation("dt(a) + lift(tau_a2) + grad(phi) - div(grad_a)/Rm - ((grad(a)@u0-u0@grad(a)) + cross(u,b0))= 0")
    problem.add_equation("dt(u) + dot(u0,grad(u)) + dot(u,grad(u0)) + grad(p) + lap(Co*(cross(a, b0)))  \
                         - div(grad_u)/Reynolds - 2*cross(omega, u) + lift(tau_u2) = 0" )
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
    solver = problem.build_solver(entry_cutoff=0)
    growth = []
    for p in solver.subproblems[3:4]:
        solver.solve_sparse(p, NEV, target=target)
        growth.append(np.max(solver.eigenvalues.real))
    return np.max(growth)
if __name__ == "__main__":

    import time
    import matplotlib.pyplot as plt
    comm = MPI.COMM_WORLD

    # Parameters
    Nx = 128
    Co = 0.08/10
    Rm = 200
    Pm = 1
    Reynolds = Rm/Pm 
    omega0 = 1
    b0 = 1
    q = 3/2
    kz_global = np.linspace(0.1, 5,16)
    ky = 1e-5
    NEV = 5

    # Compute growth rate over local wavenumbers
    kz_local = kz_global[comm.rank::comm.size]

    t1 = time.time()
    growth_local = np.array([B_max_growth_rate(Reynolds, omega0, Rm, Co, b0, q, ky, kz, Nx, NEV=NEV) for kz in kz_local])
    t2 = time.time()
    logger.info('Elapsed solve time: %f' %(t2-t1))

    # Reduce growth rates to root process
    # growth_global = np.zeros_like(kz_global)
    # growth_global[comm.rank::comm.size] = growth_local
    # if comm.rank == 0:
    #     comm.Reduce(MPI.IN_PLACE, growth_global, op=MPI.SUM, root=0)
    # else:
    #     comm.Reduce(growth_global, growth_global, op=MPI.SUM, root=0)

    # Plot growth rates from root process
    if comm.rank == 0:
        plt.figure(figsize=(6,4))
        plt.plot(kz_local, growth_local, '-',label = "max = " +str(round(max(growth_local),2))+" at "+str(kz_local[np.where(growth_local==np.max(growth_local))]) )
        plt.xlabel(r'$k_z$')
        plt.ylabel(r'$\mathrm{Re}(\sigma)$')
        plt.title(f'MRI growth rates Rm={Rm}, Pm={Pm}, Co={Co}')
        plt.tight_layout()
        # plt.xlim(0, 1.75)
        # plt.ylim(-0.09, 0.04)
        plt.legend()
        # plt.savefig(f"./plots/MRI_{Rm}_{Pm}_{Co}_{NEV}.pdf")
        plt.show()
