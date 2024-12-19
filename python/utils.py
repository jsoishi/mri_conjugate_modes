import numpy as np
import magneto_rpc_evp_b_inviscid_O19_norm as evp

def load_params(filename):
    inputs = []
    with np.load(filename) as data:
        for f in evp.evp_params._fields:
            inputs.append(data[f].item())
        params = evp.evp_params(*inputs)

    return params
