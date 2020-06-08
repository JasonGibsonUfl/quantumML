

from django.db import models
import numpy as np
class Band_Structure(models.Model):

    def __init__(self, kpoints, eigenvals, lattice, efermi, labels_dict=None,
                 coords_are_cartesian=False, structure=None, projections=None):
        """
        Args:
            kpoints: list of kpoint as numpy arrays, in frac_coords of the
                given lattice by default
            eigenvals: dict of energies for spin up and spin down
                {Spin.up:[][],Spin.down:[][]}, the first index of the array
                [][] refers to the band and the second to the index of the
                kpoint. The kpoints are ordered according to the order of the
                kpoints array. If the band structure is not spin polarized, we
                only store one data set under Spin.up
            lattice: The reciprocal lattice as a pymatgen Lattice object.
                Pymatgen uses the physics convention of reciprocal lattice vectors
                WITH a 2*pi coefficient
            efermi: fermi energy
            labels_dict: (dict) of {} this links a kpoint (in frac coords or
                cartesian coordinates depending on the coords) to a label.
            coords_are_cartesian: Whether coordinates are cartesian.
            structure: The crystal structure (as a pymatgen Structure object)
                associated with the band structure. This is needed if we
                provide projections to the band structure
            projections: dict of orbital projections as {spin: ndarray}. The
                indices of the ndarrayare [band_index, kpoint_index, orbital_index,
                ion_index].If the band structure is not spin polarized, we only
                store one data set under Spin.up.
        """
        self.efermi = efermi
        self.lattice_rec = lattice
        self.kpoints = []
        self.labels_dict = {}
        self.structure = structure
        self.projections = projections or {}
        self.projections = {k: np.array(v) for k, v in self.projections.items()}

        if labels_dict is None:
            labels_dict = {}

        if len(self.projections) != 0 and self.structure is None:
            raise Exception("if projections are provided a structure object"
                            " needs also to be given")

        for k in kpoints:
            # let see if this kpoint has been assigned a label
            label = None
            for c in labels_dict:
                if np.linalg.norm(k - np.array(labels_dict[c])) < 0.0001:
                    label = c
                    self.labels_dict[label] = Kpoint(
                        k, lattice, label=label,
                        coords_are_cartesian=coords_are_cartesian)
            self.kpoints.append(
                Kpoint(k, lattice, label=label,
                       coords_are_cartesian=coords_are_cartesian))
        self.bands = {spin: np.array(v) for spin, v in eigenvals.items()}
        self.nb_bands = len(eigenvals[Spin.up])
        self.is_spin_polarized = len(self.bands) == 2