import numpy as np
# import scipy
from matplotlib import pyplot as plt


fname = 'data/eigenmodes-Nx-128-Re-100000-Rm-100000-R-1_001-q-0_75-kz-0_447-ky-0_263_etools.npz'
with np.load(fname, allow_pickle=True) as data:
    evalues_primary = data['all_eigenvalues']
    evalues_secondary = data['all_eigenvalues_hires']
    eigenvalues = evalues_primary
    right_eigenvectors = data['all_eigenvectors']
    modified_left_eigenvectors = data['all_modified_left_eigenvectors']

finite_right_eigenvectors = right_eigenvectors[:, np.isfinite(evalues_primary)]
finite_modified_left_eigenvectors = modified_left_eigenvectors[:, np.isfinite(evalues_primary)]

inner_product_matrix = finite_modified_left_eigenvectors.T.conj()@finite_right_eigenvectors
plt.pcolormesh(np.abs(inner_product_matrix))
plt.colorbar()
plt.show()