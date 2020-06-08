import yaml
from simulation.analysis.symmetry.spacegroup import Spacegroup


def run():
    with open("simulation/data/spacegroups.yml", 'r') as stream:
        try:
            print(stream)
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    elements =[]
    for key, v in data.items():
        elements.append(key)
    print(elements)
    for el in elements:
        element = data[el]
        number = element['number']
        hm = element['hm']
        hall = element['hall']
        schoenflies = element['schoenflies']
        lattice_system = element['system']
        s1= Spacegroup(number=number, hm=hm, hall=hall, schoenflies=schoenflies, lattice_system=lattice_system)
        s1.save()
        #e1 = Element()
        print(lattice_system)
        #e1.save()
