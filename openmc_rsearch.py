import openmc
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import math
import random
import subprocess

from matplotlib import rcParams

rcParams['axes.autolimit_mode'] = 'round_numbers'
rcParams['axes.labelsize'] = 'large'
rcParams['axes.xmargin'] = 0
rcParams['axes.ymargin'] = 0
rcParams['axes.axisbelow'] = True
rcParams['font.family'] = 'serif'
rcParams['pdf.use14corefonts'] = True
rcParams['savefig.bbox'] = 'tight'
rcParams['font.size'] = 12.0
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

model = openmc.model.Model()

# --------------------------------------------------
#           M A T E R I A L
# --------------------------------------------------

m_fuel = openmc.Material(name='UO2 fuel')
m_fuel.set_density('g/cm3', 10.9)
m_fuel.add_element('U', 1.0, enrichment=10, enrichment_type='wo')
m_fuel.add_nuclide('O16', 2.0)
m_fuel.temperature = 773

m_graphite_c_buffer = openmc.Material(name='graphite buffer')
m_graphite_c_buffer.add_element('C', 1.0)
m_graphite_c_buffer.add_s_alpha_beta('c_Graphite')
m_graphite_c_buffer.set_density('g/cm3', 0.97)
m_graphite_c_buffer.temperature = 773

m_graphite_pyc = openmc.Material(name='pyc')
m_graphite_pyc.add_element('C', 1.0)
m_graphite_pyc.add_s_alpha_beta('c_Graphite')
m_graphite_pyc.set_density('g/cm3', 1.91)
m_graphite_pyc.temperature = 773

m_sic = openmc.Material(name='sic')
m_sic.add_element('C' , 0.299788082, 'wo')
m_sic.add_element('Si', 0.700211917, 'wo')
m_sic.set_density('g/cm3', 3.20)
m_sic.temperature = 773

mo_graphite = openmc.Material(name='monolith_graphite')
mo_graphite.set_density('g/cm3', 1.8)
mo_graphite.add_element('C', 1.0, 'wo')
mo_graphite.add_s_alpha_beta('c_Graphite')
mo_graphite.temperature = 773

ma_graphite = openmc.Material(name='matrix_graphite')
ma_graphite.set_density('g/cm3', 1.75)
ma_graphite.add_element('C', 1.0)
ma_graphite.add_s_alpha_beta('c_Graphite')
ma_graphite.temperature = 773

Er2O3 = openmc.Material(name='Er2O3_pois')
Er2O3.set_density('g/cm3', 8.64)
Er2O3.add_nuclide('Er162', 0.0008467223, 'wo')
Er2O3.add_nuclide('Er164', 0.0137148114, 'wo')
Er2O3.add_nuclide('Er166', 0.2915233685, 'wo')
Er2O3.add_nuclide('Er167', 0.2007568234, 'wo')
Er2O3.add_nuclide('Er168', 0.2353260926, 'wo')
Er2O3.add_nuclide('Er170', 0.1323918338, 'wo')
Er2O3.add_nuclide('O16', 0.1254403479,   'wo')
Er2O3.temperature = 773

# --------------------------------------------------
#           G E O M E T R Y
# --------------------------------------------------

# region in which TRISOs are generated
reactor_bottom = -0.5
reactor_top    = 0.5
fuel_cyl_r     = 1.0
cell_pitch     = 5.0

fuel_cyl = openmc.ZCylinder(r= fuel_cyl_r)
min_z = openmc.ZPlane(z0= reactor_bottom, boundary_type='periodic')
max_z = openmc.ZPlane(z0= reactor_top, boundary_type='periodic')

r_triso = -fuel_cyl & +min_z & -max_z

hex_boundary = openmc.model.HexagonalPrism(cell_pitch/2, 'x', boundary_type='periodic')


erb_sphere = openmc.Sphere(r=150e-4)

erb_univ = openmc.Universe(cells=[
    openmc.Cell(fill=Er2O3, region=-erb_sphere),
    openmc.Cell(fill=ma_graphite, region=+erb_sphere)
])

#-----------------------
# TRISO univ
#-----------------------
spheres = [openmc.Sphere(r=1e-4*r) for r in [250.0, 342.0, 381.0, 416.0, 450.0]]

u_triso = openmc.Universe(cells=[
    openmc.Cell(fill=m_fuel,              region= -spheres[0]),
    openmc.Cell(fill=m_graphite_c_buffer, region= +spheres[0] & -spheres[1]),
    openmc.Cell(fill=m_graphite_pyc,      region= +spheres[1] & -spheres[2]),
    openmc.Cell(fill=m_sic,               region= +spheres[2] & -spheres[3]),
    openmc.Cell(fill=m_graphite_pyc,      region= +spheres[3] & -spheres[4]),
    openmc.Cell(fill=ma_graphite,         region= +spheres[4])
])

# TRISO 
total_pf  = 30/100
pf_poison = 0.1/100
pf_triso  = total_pf - pf_poison

compact_vol = np.pi*fuel_cyl.r**2*max_z.z0*2
poison_vol  = (4/3)*np.pi*erb_sphere.r**3
triso_vol   = (4/3)*np.pi*spheres[-1].r**3
n_triso_particle  = math.ceil(pf_triso*compact_vol / triso_vol)
n_poison_particle = math.ceil(pf_poison*compact_vol / poison_vol)

fuel_centers = openmc.model.pack_spheres(radius=spheres[-1].r, region=r_triso, num_spheres=n_triso_particle, seed=12345678)

poison_centers = []

min_dist_sq = (erb_sphere.r + spheres[-1].r)**2
poison_dist_sq = (2 * erb_sphere.r)**2

def get_random_cylindrical(r_max, z_min, z_max):
    r = r_max * np.sqrt(random.random())
    theta = 2 * np.pi * random.random()
    return np.array([r * np.cos(theta), r * np.sin(theta), random.uniform(z_min, z_max)])

attempts = 0
max_attempts = 1000000

while len(poison_centers) < n_poison_particle and attempts < max_attempts:
    candidate = get_random_cylindrical(fuel_cyl_r, reactor_bottom + erb_sphere.r, reactor_top - erb_sphere.r)
    
    # fuel particles
    distances_fuel_sq = np.sum((fuel_centers - candidate)**2, axis=1)
    if np.any(distances_fuel_sq < min_dist_sq):
        attempts += 1
        continue
        
    # placed poison particles
    if poison_centers:
        distances_poison_sq = np.sum((np.array(poison_centers) - candidate)**2, axis=1)
        if np.any(distances_poison_sq < poison_dist_sq):
            attempts += 1
            continue
            
    poison_centers.append(candidate)
    attempts += 1

# Create list of TRISO objects
trisos = []
for center in fuel_centers:
    trisos.append(openmc.model.TRISO(spheres[-1].r, u_triso, center))

for center in poison_centers:
    trisos.append(openmc.model.TRISO(erb_sphere.r, erb_univ, center))


fuel_rod = openmc.Cell(region=r_triso)
lower_left, upp_right = fuel_rod.region.bounding_box
shape = (10, 10, 5)
pitch = (upp_right - lower_left)/shape
triso_latt = openmc.model.create_triso_lattice(trisos, lower_left, pitch, shape, ma_graphite)
fuel_rod.fill = triso_latt

model.geometry = openmc.Geometry(root=openmc.Universe(cells=[
    fuel_rod,
    openmc.Cell(fill=mo_graphite, region=+fuel_cyl & -hex_boundary & +min_z & -max_z)
]))

# --------------------------------------------------
#           S E T T I N G S
# --------------------------------------------------

settings = openmc.Settings()

settings.particles = int(20_000)
settings.inactive = 30
settings.batches = 110
settings.temperature={
         'method'    :'nearest',
         'tolerance' : 300.0,
 }
settings.output = {'summary': False, 'tallies': False}

hexagon_half_flat = np.sqrt(3.0) / 2.0 * cell_pitch/2
lower_left = (-cell_pitch/2, -hexagon_half_flat,  min_z.z0)
upper_right = (cell_pitch/2,   hexagon_half_flat,  max_z.z0)

source = openmc.IndependentSource(space=openmc.stats.Box(lower_left, upper_right))
settings.source = source

model.settings = settings

# plot
ax = model.geometry.root_universe.plot(width=(lower_left[0]*2, upper_right[1]*2), pixels=(500, 500), color_by='material', colors={Er2O3: 'red'})
fig = ax.figure
fig.savefig('triso.png', bbox_inches='tight')

model.run()