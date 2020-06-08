from django.db import models
import yaml
import os.path
import logging
import gzip

import numpy as np

import simulation
import simulation.custom as custom
from simulation.utils import *
from simulation.utils.rendering.line import Line
from simulation.utils.rendering.renderer import Renderer
from pymatgen.io.vasp.outputs import Vasprun, Locpot
from pymatgen.electronic_structure.bandstructure import BandStructure
logger = logging.getLogger(__name__)


class DOS(models.Model):
    """
    Electronic density of states..
    Relationships:
        | :mod:`~simulation.Entry` via entry
        | :mod:`~simulation.MetaData` via meta_data
        | :mod:`~simulation.Calculation` via calculation
    Attributes:
        | id
        | data: Numpy array of DOS occupations.
        | file: Source file.
        | gap: Band gap in eV.
    """

    #meta_data = models.ManyToManyField('MetaData')
    entry = models.ForeignKey('Entry', null=True, on_delete=models.CASCADE,)
    efermi = models.FloatField(default=0.0)
    gap = models.FloatField(blank=True, null=True, default=0)
    data = custom.NumpyArrayField(blank=True, null=True)
    file = models.CharField(max_length=128, blank=True, null=True)
    is_directBG = models.BooleanField(null=True)
    _efermi = 0.0
    band_structure = BandStructure

    class Meta:
        db_table = 'dos'
        app_label = 'simulation'

    @classmethod
    def read(cls, doscar='', efermi=0.0):
        print(doscar)
        try:
            band_structure = Vasprun(doscar[0:(len(doscar)-7)]+'/pbe_bands/vasprun.xml').get_band_structure()
            print(band_structure.get_band_gap())
            dos = DOS(file=doscar)
            dos._efermi = 0.0
            dos.read_doscar(dos.file)
            dos.efermi = band_structure.efermi
            gap = band_structure.get_band_gap()
            dos.gap = gap['energy']
            dos.is_directBG = gap['direct']

        except ValueError:
            raise simulation.analysis.vasp.calculation.VaspError('Could not parse DOSCAR')
        return dos

    @property
    def efermi(self):
        return self._efermi

    @efermi.setter
    def efermi(self, efermi):
        """Set the Fermi level."""
        ef = efermi - self._efermi
        self._efermi = efermi
        try:
            import ast

            self.data = np.array(self.data)
            self.site_dos = np.array(self.site_dos)
            self.data[0][:] = self.data[0][:] + ef
            self.site_dos[:][0][:] = self.site_dos[:][0][:] + ef

        except IndexError:
            pass

    @property
    def energy(self):
        """Return the array with the energies."""
        return self.data[0, :]

    def find_vbm(self, tol=1e-3):
        self.vbm = self.band_structure.get_vbm()['energy']
        return self.vbm

    def find_cbm(self, tol=1e-3):
        self.cbm = self.band_structure.get_cbm()['energy']
        return self.cbm

    def metal_find_vbm(self, tol=1e-3):
        i = (np.abs(self.energy)).argmin()
        if self.energy[i] < 0:
            i += 1
        edos = self.total_dos
        while edos[i] > tol:
            i += 1
        else:
            return self.energy[i]

    def find_gap(self, tol=1e-3):
        #self.gap = self.band_structure.get_band_gap()['energy']
        return self.gap

    def is_direct(self):
        return self.is_directBG
    _plot = None

    @property
    def plot(self):
        if self._plot is None:
            line1 = Line(zip(self.energy, self.total_dos), fill=True)
            line2 = Line([[0, 0], [0, max(self.total_dos) * 1.1]])
            canvas = Renderer(lines=[line1, line2])
            canvas.yaxis.max = max(self.total_dos) * 1.1
            canvas.yaxis.min = 0
            canvas.xaxis.label = "Energy (eV)"
            canvas.yaxis.label = "# of States/eV/unit cell"  # [mohan]
            self._plot = canvas
        return self._plot



    def get_projected_dos(self, strc, element, orbital=None, debug=False):
        """
        Get the density of states for a certain element

        Returns an NDOSx1 array with DOS for the chosen element
            and orbital type

        strc: simulation.materials.Structure
            Structure associated with this DOS
        element: str
            Symbol of the element
        orbital: str or list or None
            Which orbitals to retrieve. See site_dos for options.
            Use None to integrate all orbitals
            If you specify an orbital without phase factors or spins,
                this operation will automatically sum all bands that
                match that criteria (ex: if you specify 'd+', this
                may sum the dxy+, dyz+, dz2+, ... orbitals. Another
                example, if you put "+" it will get only the positive band
        """

        # Check for input errors
        if strc.natoms != self._site_dos.shape[0]:
            raise Exception('Structure has different atom count than DOS')
        if not element in strc.composition.comp.keys():
            raise Exception('Element not in structure')

        # Get the list of atoms to sum over
        atoms = [i for i, atom in enumerate(strc.atoms) if atom.element.symbol == element]

        # Get the number of different bands
        n = self._site_dos.shape[1]

        # If only a string is passed for orbital, convert it to list
        if isinstance(orbital, str):
            orbital = [orbital]

        # Get the list of orbitals
        if n == 4:
            all_orbs = set(['s', 'p', 'd'])
        elif n == 7:
            all_orbs = set(['s+', 's-', 'p+', 'p-', 'd-', 'd+'])
        elif n == 10 or n == 37:  # 37 == non-collinear
            all_orbs = set(['s', 'px', 'py', 'pz', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2'])
        elif n == 19:
            all_orbs = set(['s+', 'px+', 'py+', 'pz+', 'dxy+', 'dyz+', 'dz2+', 'dxz+', 'dx2+',
                            's-', 'px-', 'py-', 'pz-', 'dxy-', 'dyz-', 'dz2-', 'dxz-', 'dx2-'])
        else:
            raise Exception('Unrecognized number of columns in DOS: %d' % n)

        # Get the ones that the user wants
        if orbital is None:
            orb_to_sum = all_orbs
        else:
            orb_to_sum = set()
            for orb in orbital:
                if len(orb) == 2 and ('-' in orb or '+' in orb):
                    shl = orb[0]
                    spn = orb[1]
                    for o in all_orbs:
                        if shl in o and spn in o:
                            orb_to_sum.add(o)
                else:
                    for o in all_orbs:
                        if orb in o: orb_to_sum.add(o)

        # Print out info, if debug mode
        if debug:
            print
            "Summing orbitals: ", " ".join(orb_to_sum)
            print
            "For atoms: ", " ".join([str(x) for x in atoms])

        # Do the sum
        output = np.zeros(self.data.shape[1])
        for atom in atoms:
            for orb in orb_to_sum:
                output += self.site_dos(atom, orb)
        return output

    def site_dos(self, atom, orbital):
        """Return an NDOSx1 array with dos for the chosen atom and orbital.

        atom: int
            Atom index
        orbital: int or str
            Which orbital to plot

        If the orbital is given as an integer:
        If spin-unpolarized calculation, no phase factors:
        s = 0, p = 1, d = 2
        Spin-polarized, no phase factors:
        s-up = 0, s-down = 1, p-up = 2, p-down = 3, d-up = 4, d-down = 5
        If phase factors have been calculated, orbitals are
        s, py, pz, px, dxy, dyz, dz2, dxz, dx2
        double in the above fashion if spin polarized.

        """
        # Integer indexing for orbitals starts from 1 in the _site_dos array
        # since the 0th column contains the energies
        if isinstance(orbital, int):
            return self._site_dos[atom, orbital + 1, :]
        n = self._site_dos.shape[1]
        if n == 4:
            norb = {'s': 1, 'p': 2, 'd': 3}
        elif n == 7:
            norb = {'s+': 1, 's-up': 1, 's-': 2, 's-down': 2,
                    'p+': 3, 'p-up': 3, 'p-': 4, 'p-down': 4,
                    'd+': 5, 'd-up': 5, 'd-': 6, 'd-down': 6}
        elif n == 10 or n == 37:
            norb = {'s': 1, 'py': 2, 'pz': 3, 'px': 4,
                    'dxy': 5, 'dyz': 6, 'dz2': 7, 'dxz': 8,
                    'dx2': 9}
            if n == 37:  # Non-collinear
                for k in norb.keys():  # Add 3 new columns between each entry
                    norb[k] = (norb[k] - 1) * 3 + norb[k]
        elif n == 19:
            norb = {'s+': 1, 's-up': 1, 's-': 2, 's-down': 2,
                    'py+': 3, 'py-up': 3, 'py-': 4, 'py-down': 4,
                    'pz+': 5, 'pz-up': 5, 'pz-': 6, 'pz-down': 6,
                    'px+': 7, 'px-up': 7, 'px-': 8, 'px-down': 8,
                    'dxy+': 9, 'dxy-up': 9, 'dxy-': 10, 'dxy-down': 10,
                    'dyz+': 11, 'dyz-up': 11, 'dyz-': 12, 'dyz-down': 12,
                    'dz2+': 13, 'dz2-up': 13, 'dz2-': 14, 'dz2-down': 14,
                    'dxz+': 15, 'dxz-up': 15, 'dxz-': 16, 'dxz-down': 16,
                    'dx2+': 17, 'dx2-up': 17, 'dx2-': 18, 'dx2-down': 18}
        return self._site_dos[atom, norb[orbital.lower()], :]

    @property
    def dos(self):
        if self.data.shape[0] == 3:
            return np.array([self.data[1][:]])  # make output consistent (nested list)
        elif self.data.shape[0] == 5:
            return self.data[1:3][:]

    @property
    def total_dos(self):
        if self.data.shape[0] == 3:
            return self.data[1][:]
        elif self.data.shape[0] == 5:
            return np.sum(self.data[1:3][:], 0)

    @property
    def integrated_dos(self):
        if self.data.shape[0] == 3:
            return self.data[2][:]
        elif self.data.shape[0] == 5:
            return self.data[3:5, :]

    def read_doscar(self, fname="DOSCAR"):
        """Read a VASP DOSCAR file"""
        if os.path.getsize(fname) < 300:
            return
        if os.path.splitext(fname)[1] == '.gz':
            f = gzip.open(fname, 'rb')
        else:
            f = open(fname, 'r')
        natoms = int(f.readline().split()[0])
        [f.readline() for nn in range(4)]  # Skip next 4 lines.
        # First we have a block with total and total integrated DOS
        ndos, efermi = f.readline().split()[2:4]
        self.efermi = float(efermi)
        ndos = int(ndos)
        dos = []
        for nd in range(ndos):
            dos.append(np.array([float(x) for x in f.readline().split()]))
        self.data = np.array(dos).T
        # Next we have one block per atom, if INCAR contains the stuff
        # necessary for generating site-projected DOS
        dos = []
        for na in range(natoms):
            line = f.readline()
            if line == '':
                # No site-prjected DOS
                break
            ndos = int(line.split()[2])
            line = f.readline().split()
            cdos = np.empty((ndos, len(line)))
            cdos[0] = np.array(line)
            for nd in range(1, ndos):
                line = f.readline().split()
                cdos[nd] = np.array([float(x) for x in line])
            dos.append(cdos.T)
        self._site_dos = np.array(dos)
        f.close()
