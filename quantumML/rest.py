from urllib.request import urlopen
import json
import urllib
import os
import io
import math
from pymatgen.core.structure import Structure
from ase.io import vasp
from dscribe.descriptors import SOAP
from pymatgen.io.vasp import Xdatcar, Oszicar
from sklearn.cluster import KMeans
import numpy as np

class MWRester(object):
    results = {}

    def __init__(self, api_key=None,
                 endpoint="http://2dmaterialsweb.org/rest/calculation/?"):
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = ""
        self.preamble = endpoint
        import requests
        self.session = requests.Session()
        self.session.headers = {"x-api-key": self.api_key}

    def __str__(self):
        return '%s' % self.results

    def _make_request(self, sub_url, payload=None, method="GET",
                      mp_decode=True):
        url = self.preamble + sub_url + "/" + self.api_key
        x = urlopen(url)

        response = self.session.get(url, verify=False)
        data = json.loads(response.text)
        return data

    def get_calculation(self, band_gap_range=None, formation_energy_range=None, elements=[], space_group_number=None,
                        dimension=None, crystal_system=None, name=None):
        '''
        Method to that queries materialsweb database and returns a list of dictionaries of calculations that
        fit the querries parameters. Additionally
        Parameters:
            band_gap_range (list): List of band gap range e.g. [min, max]
            formation_energy_range (list): List of formation energy range e.g. [min, max]
            elements (list): List of str of elements
            space_group_number (int): space group number
            dimension (int): dimension as int e.g. 1 2 3
            crystal_system (str): name of crystal system
            name (str): name of material e.g. MoS2
        Returns:
            results: List of results matching quire parameters
        '''

        suburl = ''
        if band_gap_range != None:
            suburl += 'band_gap_min=' + str(band_gap_range[0]) + '&band_gap_max=' + str(band_gap_range[-1]) + '&'
        if formation_energy_range != None:
            suburl += 'formation_energy_min=' + str(formation_energy_range[0]) + '&formation_ener_max=' + str(
                formation_energy_range[-1]) + '&'

        if (space_group_number != None):
            suburl += 'spacegroup_number=' + str(space_group_number) + '&'

        if (dimension != None):
            suburl += 'dimension=' + str(dimension) + '&'

        if (crystal_system != None):
            'lattice_system=' + str(crystal_system) + '&'
        self.results = self._make_request(suburl)['results']
        return self.results

    def as_pymatgen_struc(self):
        '''
        Method that converts results to list of pymatgen strucutures
        Returns:
             struc: List of pymatgen structures
        '''
        struc = []
        for c in self.results:
            urlp = ('http://2d' + c['path'][9:21] + '.org/' + c['path'][22:] + '/POSCAR')
            file = urllib.request.urlopen(urlp)
            poscar = ''
            for line in file:
                poscar += line.decode("utf-8")

            s = Structure.from_str(poscar, fmt='poscar')
            struc.append(s)

        return struc

    '''Write files'''
    
    def write(self,index=0):
        '''
        Writes INCAR, KPOINTS, POSCAR of entry to current directory

        Parameters:
            index (int): index of entry to write files for
        '''
        self.write_poscar(index=index)
        self.write_incar(index=index)
        self.write_kpoints(index=index)

    def write_all(self=0):
        '''
        Creates a directory named by composition for every entry in results. Then, Writes INCAR, KPOINTS,
        POSCAR of entry to respective directory
        '''
        for index in range(0, len(self.results)):
            dir_name = self.results[index]['composition'].split('/')[-2].replace('%', '')
            os.mkdir(dir_name)
            os.chdir(dir_name)
            self.write_poscar(index=index)
            self.write_incar(index=index)
            self.write_kpoints(index=index)
            os.chdir('..')

    def write_poscar(self,index=0):
        '''
        Writes POSCAR of entry to current directory

        Parameters:
            index (int): index of entry to write files for
        '''
        urlp = 'http://2dmaterialsweb.org/'+ self.results['results'][index]['path'][22:] + '/POSCAR'
        file = urllib.request.urlopen(urlp)
        with open('POSCAR','a') as poscar:
            for line in file:
                decoded_line = line.decode("utf-8")
                poscar.write(decoded_line)

    def write_kpoints(self,index=0):
        '''
        Writes KPOINTS of entry to current directory

        Parameters:
            index (int): index of entry to write files for
        '''
        urlp = 'http://2dmaterialsweb.org/'+ self.results[index]['path'][22:] + '/KPOINTS'
        file = urllib.request.urlopen(urlp)
        with open('KPOINTS','a') as poscar:
            for line in file:
                decoded_line = line.decode("utf-8")
                poscar.write(decoded_line)

    def write_incar(self,index=0):
        '''
        Writes INCAR of entry to current directory

        Parameters:
            index (int): index of entry to write files for
        '''
        urlp = 'http://2dmaterialsweb.org/'+ self.results[index]['path'][22:] + '/INCAR'
        file = urllib.request.urlopen(urlp)
        with open('INCAR','a') as poscar:
            for line in file:
                decoded_line = line.decode("utf-8")
                poscar.write(decoded_line)

    @staticmethod
    def get_KRR(system):
        '''
        Method to allow easy access to all pre-trainned kernal ridge regresion machine learning models of GASP runs
        Args:
            system (str): A chemical system (e.g. Cd-Te)
        returns:
            pickle object of machine learning model
        '''
        import pickle
        urlm ='http://2dmaterialsweb.org/static/models/'+system+'.sav'
        print(urlm)
        model = pickle.load(urllib.request.urlopen(urlm))
        return model
    
    '''Machine Learning'''
    
    @staticmethod
    def prep_ml_formation_energy(calculation, fileroot='.'):
        os.mkdir('formation_energy')
        os.chdir('formation_energy')
        urlo = ('http://2d' + calculation['path'][9:21] + '.org/' + calculation['path'][22:] + '/OSZICAR')
        fileo = urllib.request.urlopen(urlo)
        urlx = ('http://2d' + calculation['path'][9:21] + '.org/' + calculation['path'][22:] + '/XDATCAR')
        filex = urllib.request.urlopen(urlx)
        with open('OsZICAR','a') as oszicar:
            for line in fileo:
                decoded_line = line.decode("utf-8")
                oszicar.write(decoded_line)

        with open('XdATCAR','a') as xdatcar:
            for line in filex:
                decoded_line = line.decode("utf-8")
                xdatcar.write(decoded_line)

        n = 100  # number of steps to sample
        s_extension = 'poscar'
        e_extension = 'energy'
        prefix = ''  # prefix for files, e.g. name of structure
        # e.g. "[root]/[prefix][i].[poscar]" where i=1,2,...,n
        s_list = Xdatcar('XdATCAR').structures
        e_list = [step['E0'] for step in Oszicar('OsZICAR').ionic_steps]
        if n < len(s_list) - 1:
            # the idea here is to obtain a subset of n energies
            # such that the energies are as evenly-spaced as possible
            # we do this in energy-space not in relaxation-space
            # because energies drop fast and then level off
            idx_to_keep = []
            fitting_data = np.array(e_list)[:, np.newaxis]  # kmeans expects 2D
            kmeans_model = KMeans(n_clusters=n)
            kmeans_model.fit(fitting_data)
            cluster_centers = sorted(kmeans_model.cluster_centers_.flatten())
            for centroid in cluster_centers:
                closest_idx = np.argmin(np.subtract(e_list, centroid) ** 2)
                idx_to_keep.append(closest_idx)
            idx_to_keep[-1] = len(e_list) - 1  # replace the last
            idx_list = np.arange(len(s_list))
            idx_batched = np.array_split(idx_list[:-1], n)
            idx_kept = [batch[0] for batch in idx_batched]
            idx_kept.append(idx_list[-1])
        else:
            idx_kept = np.arange(len(e_list))

        for j, idx in enumerate(idx_kept):
            filestem = str(j)
            s_filename = '{}/{}{}.{}'.format(fileroot, prefix, filestem, s_extension)
            e_filename = '{}/{}{}.{}'.format(fileroot, prefix, filestem, e_extension)
            s_list[idx].to(fmt='poscar', filename=s_filename)
            with open(e_filename, 'w') as f:
                f.write(str(e_list[idx]))


    @staticmethod
    def get_soap(calculation, rcut=7, nmax=6, lmax=8, fmt='MW'):
        if fmt == 'MW':
            urlp = 'http://2dmaterialsweb.org/' + calculation['path'][22:] + '/POSCAR'
            file = urllib.request.urlopen(urlp)
            file = file.read().decode("utf-8")
            file = io.StringIO(file)
        elif fmt == 'poscar':
            file = calculation
        ml=vasp.read_vasp(file)
        periodic_soap = SOAP(
            periodic=True,
            species=np.unique(ml.get_atomic_numbers()),
            rcut=rcut,
            nmax=nmax,
            lmax=lmax,
            rbf='gto',
            sigma=0.125,
            average=True
        )
        soap = periodic_soap.create(ml)
        #soap = 1
        return soap

    @staticmethod
    def cut_off_function(r, R_c):
        if r <= R_c:
            f_c = (math.cos(math.pi * r / R_c) + 1) * .5
        else:
            f_c = 0
        return f_c

    @staticmethod
    def radial_symmetry_function(eta, R_s, r, R_c, species=None):
        summ = 0
        minr = min(r)
        if species == None:
            for r_ij in r:
                if r_ij > minr:
                    summ = summ + math.exp(-eta * ((r_ij - R_s) ** 2)) * MWRester.cut_off_function(r_ij, R_c)
        else:
            for r_ij, z in zip(r, species):
                if r_ij > minr:
                    summ = summ + math.exp(-eta * ((r_ij - R_s) ** 2)) * MWRester.cut_off_function(r_ij, R_c) * z
        return summ

    @staticmethod
    def angular_symmetry_function_ps(epsi, lamda, theta, eta, R_c, r_ij, r_ik, r_jk, ):
        g_2 = ((1 + lamda * math.cos(theta)) ** epsi) * math.exp(-eta * (r_ij ** 2 + r_ik ** 2 + r_jk ** 2)) \
              * MWRester.cut_off_function(r_ij, R_c) * MWRester.cut_off_function(r_ik, R_c) * MWRester.cut_off_function(r_jk, R_c)
        return g_2

    @staticmethod
    def get_symmetry_functions_g1(structure, R_c=6, R_s=3, eta=1, weighted=False):
        '''
        calculates radial symmetry functions described by equation 4 in:
        Behler, J., &amp; Parrinello, M. (2007). Generalized Neural-Network Representation
        of High-Dimensional Potential-Energy Surfaces. Physical Review Letters, 98(14).

        Parameters:
            structure: pymatgen structure of material to calculate 
            R_c (float, optional): cut of radius
            R_s (float, optional): gaussian parameter
            eta (float, optional): gaussian parameter
            weighted (bool): If true values will be weighted by the method described in ""
        return:
            g_1 (list): List of values of radial symmetry function for each atom in structure
        '''
        g_1 = []
        #structure = Structure.from_file(path + '/POSCAR')
        atom_sphere_list = []
        for a in structure.get_primitive_structure():
            coord = (a.coords)
            atom_sphere_list.append(structure.get_sites_in_sphere(coord, R_c, include_image=True))
            # print(len(b))

        for atom_sphere in atom_sphere_list:

            r_list = np.array(atom_sphere)[:, 1]
            if weighted is True:
                species_list = (np.array(atom_sphere)[:, 0])
                species = [sp.species.elements[0].Z for sp in species_list]
                g_1.append(MWRester.radial_symmetry_function(eta, R_s, r_list, R_c, species=species))
            else:

                g_1.append(MWRester.radial_symmetry_function(eta, R_s, r_list, R_c))

        return g_1
    @staticmethod
    def get_symmetry_functions_g2(structure, R_c=6, epsi=1, lamda=1, eta=1, weighted=False):
        '''
        calculates angular symmetry functions described by equation 5 in:
        Behler, J., &amp; Parrinello, M. (2007). Generalized Neural-Network Representation
        of High-Dimensional Potential-Energy Surfaces. Physical Review Letters, 98(14).

        Parameters:
            structure: pymatgen structure of material to calculate
            R_c (float, optional): cut of radius
            epsi (float, optional):
            lamda (int, optional):
            eta (float, optional): gaussian parameter
            weighted (bool): If true values will be weighted by the method described in ""
        return:
            g_2 (list): List of values of angular symmetry function for each atom in structure
        '''
        g_2 = []
        atom_sphere_list = []
        for a in structure.get_primitive_structure():
            coord = (a.coords)
            atom_sphere_list.append(structure.get_sites_in_sphere(coord, R_c, include_image=True))
            # print(len(b))

        for atom_sphere in atom_sphere_list:
            summ = 0
            r_list = np.array(atom_sphere)[:, 1]
            i = np.where(r_list == min(r_list))[0][0]
            atom_sphere = np.array(atom_sphere)
            # print(atom_sphere[0])
            for j in range(0, len(atom_sphere)):
                if i != j:
                    zj = atom_sphere[j, 0].species.elements[0].Z
                    rij = atom_sphere[i, 0].coords - atom_sphere[j, 0].coords
                    rijn = np.linalg.norm(rij)
                    for k in range(0, len(atom_sphere)):
                        if i != k and j != k:
                            rik = atom_sphere[i, 0].coords - atom_sphere[k, 0].coords
                            rjk = atom_sphere[j, 0].coords - atom_sphere[k, 0].coords
                            rikn = np.linalg.norm(rik)
                            rjkn = np.linalg.norm(rjk)
                            theta = np.dot(rij, rik) / (rikn * rijn)
                            if weighted is False:
                                summ = summ + MWRester.angular_symmetry_function_ps(epsi, lamda, theta, \
                                                                                eta, R_c, rijn, rikn, rjkn, )
                            else:
                                zk = atom_sphere[k, 0].species.elements[0].Z
                                summ = summ + MWRester.angular_symmetry_function_ps(epsi, lamda, theta, \
                                                                                eta, R_c, rijn, rikn, rjkn, ) * zj * zk
            g_2.append(summ)

        return g_2
