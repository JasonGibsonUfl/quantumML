import yaml
from simulation.analysis.symmetry.spacegroup import WyckoffSite, Spacegroup

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
    for el in elements:
        sg = data[el]
        sgn =sg['number']
        spacegroup = Spacegroup.get(sgn)
        ws = sg['wyckoff_sites']

        print(len(ws))
        for ss in ws:
            string = (str(ws[ss]['coordinate'])).split()
            x = string[0]
            y = string[1]
            z = string[2]
            multiplicity = ws[ss]['multiplicity']
            symbol =ss
            ws1 = WyckoffSite(spacegroup=spacegroup, symbol=symbol, x=x, y=y, z=z, multiplicity=multiplicity)
            ws1.save()
