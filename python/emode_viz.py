import dedalus.public as d3
import numpy as np
from utils import load_params
from bokeh.io import curdoc
from bokeh.layouts import column,layout, row
from bokeh.models import ColumnDataSource, CustomJS, Dropdown
from bokeh.plotting import figure

import magneto_rpc_evp_b_inviscid_O19_norm as evp
import sys
import logging
logger = logging.getLogger(__name__)

filename = "/home/jsoishi/vc/mri_conjugate_modes/python/data/eigenmodes-Nx-128-Re-10000-Rm-10000-R-1_001-q-0_75-kz-0_447-ky-0_263_etools.npz"
params = load_params(filename)
EVP = evp.dense_evp(params, reload=True)
grid_select = (0,0) # index only radius for now. Assume 3D!
field_menu = []
field_indices = {}
field_count = 0
for i,f in enumerate(EVP.solver.problem.variables):
    field_name = f.name
    tensor_rank = len(f.tensorsig)
    if tensor_rank == 0:
        field_menu.append((f.name, f.name))
        field_indices[f.name] = (i, ())
    elif tensor_rank == 1:
        for j,cs in enumerate(f.tensorsig[0]):
            field_name = f"{f.name}_{cs}"
            field_menu.append((field_name,field_name))
            field_indices[field_name] = (i,(j,))
    elif tensor_rank == 2:
        for j, cs in enumerate(f.tensorsig[0]):
            for k, cs2 in enumerate(f.tensorsig[1]):
                field_name = f"{f.name}_{cs}_{cs2}"
                field_menu.append((field_name,field_name))
                field_indices[field_name] = (i,(j,k))
logger.info(f"field_menu = {field_menu}")                

du = EVP.solver.problem.variables[0]
dist = EVP.solver.dist
x, = du.domain.bases[2].global_grids(dist=dist, scales=(1,))

c = EVP.evalues
c_p = EVP.evalues_primary
c_s = EVP.evalues_secondary
eigenvalues = EVP.evalues
logger.info(f"{len(eigenvalues)} good modes; {len(c_p)} total modes")

spectrum = figure(height=400, width=500, title='Spectrum', tools='tap,box_zoom,wheel_zoom, pan')
spectrum.xaxis.axis_label = "c (real)"
spectrum.yaxis.axis_label = "c (imag)"
spectrum_source = ColumnDataSource(data=dict(x=c.real, y=c.imag))
#spectrum_source_s = ColumnDataSource(data=dict(x=c_s.real, y=c_s.imag))
spectrum.scatter('x', 'y', source=spectrum_source, size=10, alpha=0.4)
#spectrum.scatter('x', 'y', source=spectrum_source_s, marker='cross',size=10, alpha=0.4,color='orange')

# initialize eigenmode plot
emode_index = None
field_index_key = field_menu[0][0]
init_data = {'x': [0,], 'y':[0]}
emode_source_real = ColumnDataSource(init_data)
emode_source_imag = ColumnDataSource(init_data)
emode = figure(height=400, width=500, title='Eigenmode')
emode.xaxis.axis_label = "radius"
emode.line('x', 'y', source=emode_source_real, legend_label="cos")
emode.scatter('x', 'y', source=emode_source_real, fill_color="white", size=8, alpha=0.4)
emode.line('x', 'y', source=emode_source_imag, line_dash="dashed", color='orange', legend_label="sin")
emode.scatter('x', 'y', source=emode_source_imag, fill_color="orange", size=8, alpha=0.4)

def select_field(field_name):
    for f in EVP.solver.problem.variables:
        if f.name == field_name:
            return f
def set_state(mode_index, field_index):
    EVP.solver.set_state(mode_index)
    u = select_field('u')
    ux = u['g'][2]
    normalization = np.max(ux)
    mode = EVP.solver.problem.variables[field_index]
    mode['g'] /= normalization
    return mode

def emode_select(mode_index, field_index_key):

    field_index = field_indices[field_index_key][0]
    tensor_component = field_indices[field_index_key][1]
    total_index = tensor_component + grid_select
    if mode_index is None:
        emode_source_real.data = init_data
        emode_source_imag.data = init_data
        emode.title.text = "Eigenmode"
    else:
        mode = set_state(mode_index, field_index)
        print(f"mode = {mode.name}")
        print(f"evalue = {EVP.evalues_primary[mode_index]}")
        b = select_field('b')
        div_b2 = d3.div(b)**2
        Ly = 2*np.pi/params.ky
        Lz = 2*np.pi/params.kz
        avg_div_b = d3.Integrate(div_b2).evaluate()
        print(f"mean div(b) = {np.sqrt(avg_div_b['g'][0,0,0])/Lz/Ly}")
        u = select_field('u')
        div_u2 = d3.div(u)**2
        avg_div_u = d3.Integrate(div_u2).evaluate()
        print(f"mean div(u) = {np.sqrt(avg_div_u['g'][0,0,0])/Lz/Ly}")

        emode_source_real.data = {'x':x.ravel(), 'y': mode['g'][total_index].real}
        emode_source_imag.data = {'x':x.ravel(), 'y': mode['g'][total_index].imag}
        emode.title.text = field_index_key


def change_mode_callback(attr, old, new):
    global emode_index
    try:
        raw_index = new[0]
        emode_index = EVP.evalues_index[raw_index]
    except IndexError:
        emode_index = None
    emode_select(emode_index, field_index_key)

def change_field_callback(event):
    global field_index_key
    field_index_key = event.item
    emode_select(emode_index, field_index_key)

spectrum_source.selected.on_change("indices", change_mode_callback)    
dropdown = Dropdown(label="Field", button_type="warning", menu=field_menu)
dropdown.on_event("menu_item_click", change_field_callback)

curdoc().add_root(layout([row(dropdown), row(spectrum, emode)]))
