import yaml
from simulation.materials.element import Element

def run():
	with open("simulation/data/elements/data.yaml", 'r') as stream:
	    try:
		    print(stream)
		    data = yaml.safe_load(stream)
	    except yaml.YAMLError as exc:
		    print(exc)

	with open("simulation/data/elements/chemical_potentials.yml", 'r') as stream:
	    try:
		    print(stream)
		    data2 = yaml.safe_load(stream)
	    except yaml.YAMLError as exc:
		    print(exc)

	elements =[]
	for key, v in data.items():
	    elements.append(key)
	print(elements)
	for el in elements:
	    element = data[el]
	    z = element['z']
	    name = element['name']
	    symbol = el

	    group = element['group']
	    period =element['period']

	    mass = element['mass']
	    density = element['density']
	    volume = element['volume']
	    atomic_radii = element['atomic_radii']
	    van_der_waals_radii = element['van_der_waals_radii']
	    covalent_radii = element['covalent_radii']
	    try:
		    scattering_factors = element['scattering_factors']
	    except:
		    scattering_factors = {}

	    melt = element['melt']
	    boil = element['boil']
	    specific_heat = element['specific_heat']

	    electronegativity = element['electronegativity']
	    first_ionization_energy = element['first_ionization_energy']
	    s_elec = element['s_elec']
	    p_elec = element['p_elec']
	    d_elec = element['d_elec']
	    f_elec = element['f_elec']

	    try:
		    HHI_P = element['HHI_P']
	    except:
		    HHI_P = 0

	    try:
		    HHI_R = element['HHI_R']

	    except:
		    HHI_R = 0
	    production = element['production']
	    try:
	        chemical_potential = data2['standard']['elements'][el]
	    except:
	        chemical_potential = 0

	    e1 = Element(z=z, name=name, symbol=symbol, group=group, period=period, mass=mass, density=density,
	                 volume=volume, atomic_radii=atomic_radii, van_der_waals_radii=van_der_waals_radii,
	                 covalent_radii=covalent_radii, scattering_factors=scattering_factors,chemical_potential=chemical_potential, melt=melt, boil=boil,
	                 specific_heat=specific_heat, electronegativity=electronegativity,
	                 first_ionization_energy=first_ionization_energy, s_elec=s_elec,
	                 p_elec=p_elec, d_elec=d_elec, f_elec=f_elec, HHI_P=HHI_P, HHI_R=HHI_R, production=production)
	    print(e1)
	    e1.save()
