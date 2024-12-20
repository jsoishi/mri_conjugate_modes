import h5py
import matplotlib.pyplot as plt
import sys

datadir = sys.argv[-1]
filename = f"{datadir}/scalars/scalars_s1.h5"
error_names = ['|div_u|', '|div_b|', '|grad_phi|']
energy_names = ['KE', 'ME']
with h5py.File(filename, 'r') as df:
    plt.subplot(211)
    for f in energy_names:
        t = df[f'tasks/{f}'].dims[0][0][:]
        f_data = df[f'tasks/{f}'][:,0,0,0]
        plt.semilogy(t, f_data,label=f"{f}")

    plt.xlabel("time")
    plt.ylabel("Energy")
    plt.legend()
    plt.subplot(212)
    for f in error_names:
        t = df[f'tasks/{f}'].dims[0][0][:]
        f_data = df[f'tasks/{f}'][:,0,0,0]
        plt.semilogy(t, f_data,label=f"{f}")
    plt.xlabel("time")
    plt.ylabel("Error")
    plt.legend()

plt.tight_layout()
plt.savefig(datadir+"/energies_errors.png", dpi=300)
