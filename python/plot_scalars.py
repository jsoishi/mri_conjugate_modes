import h5py
import matplotlib.pyplot as plt
import sys

datadir = sys.argv[-1]
filename = f"{datadir}/scalars/scalars_s1.h5"
#field_names = ['|div_u|', '|div_b|']
field_names = ['KE', 'ME']
with h5py.File(filename, 'r') as df:
    for f in field_names:
        t = df[f'tasks/{f}'].dims[0][0][:]
        f_data = df[f'tasks/{f}'][:,0,0,0]
        plt.plot(t, f_data, 'x-',label=f"{f}")
plt.legend()

plt.savefig(datadir+"/energies.png", dpi=300)
#plt.savefig(datadir+"/div_u_div_b.png", dpi=300)
