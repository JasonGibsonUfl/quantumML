from pymatgen.io.vasp.sets import DictSet, _load_yaml_config
from pymatgen.core.structure import Structure
from atomate.vasp.workflows.base.core import get_wf
from atomate.vasp.powerups import add_small_gap_multiply, add_stability_check, add_modify_incar, \
    add_wf_metadata, add_common_powerups


from atomate.vasp.config import SMALLGAP_KPOINT_MULTIPLY, STABILITY_CHECK, VASP_CMD, DB_FILE, \
    ADD_WF_METADATA, VDW_KERNEL_DIR
import numpy as np
import os
from monty.serialization import loadfn
from pathlib import Path

module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

def _read_user_incar(fname):
    fpath = os.path.join(module_dir,fname)
    fil = open(fpath, 'r')
    lines = fil.readlines()
    incar = {}
    for line in lines:
        key = (line.split('=')[0].strip())
        val = line.split('=')[-1].strip()
        incar[key] = val
    return incar



class MPRelaxSet2D(DictSet):
    """
    Implementation of VaspInputSet utilizing parameters in the public
    Materials Project. Typically, the pseudopotentials chosen contain more
    electrons than the MIT parameters, and the k-point grid is ~50% more dense.
    The LDAUU parameters are also different due to the different psps used,
    which result in different fitted values.
    """

    CONFIG = _load_yaml_config("MPRelaxSet")

    def __init__(self, structure, **kwargs):
        """
        :param structure: Structure
        :param kwargs: Same as those supported by DictSet.
        """
        incar = _read_user_incar('Relax2D.txt')
        super().__init__(structure, MPRelaxSet2D.CONFIG, user_incar_settings=incar, **kwargs)
        self.kwargs = kwargs

class MPScanRelaxSet2D(DictSet):
    """
    Class for writing a relax input set using Strongly Constrained and
    Appropriately Normed (SCAN) semilocal density functional.
    Notes:
        1. This functional is only available from VASP.5.4.3 upwards.
        2. Meta-GGA calculations require POTCAR files that include
        information on the kinetic energy density of the core-electrons,
        i.e. "PBE_52" or "PBE_54". Make sure the POTCARs include the
        following lines (see VASP wiki for more details):
            $ grep kinetic POTCAR
            kinetic energy-density
            mkinetic energy-density pseudized
            kinetic energy density (partial)
    """

    CONFIG = _load_yaml_config("MPSCANRelaxSet")

    def __init__(self, structure, bandgap=0, **kwargs):
        """
        Args:
            structure (Structure): Input structure.
            bandgap (int): Bandgap of the structure in eV. The bandgap is used to
                    compute the appropriate k-point density and determine the
                    smearing settings.
                    Metallic systems (default, bandgap = 0) use a KSPACING value of 0.22
                    and Methfessel-Paxton order 2 smearing (ISMEAR=2, SIGMA=0.2).
                    Non-metallic systems (bandgap > 0) use the tetrahedron smearing
                    method (ISMEAR=-5, SIGMA=0.05). The KSPACING value is
                    calculated from the bandgap via Eqs. 25 and 29 of Wisesa, McGill,
                    and Mueller [1] (see References). Note that if 'user_incar_settings'
                    or 'user_kpoints_settings' override KSPACING, the calculation from
                    bandgap is not performed.
            vdw (str): set "rVV10" to enable SCAN+rVV10, which is a versatile
                    van der Waals density functional by combing the SCAN functional
                    with the rVV10 non-local correlation functional. rvv10 is the only
                    dispersion correction available for SCAN at this time.
            **kwargs: Same as those supported by DictSet.
        References:
            [1] P. Wisesa, K.A. McGill, T. Mueller, Efficient generation of
            generalized Monkhorst-Pack grids through the use of informatics,
            Phys. Rev. B. 93 (2016) 1–10. doi:10.1103/PhysRevB.93.155109.
        """
        incar = _read_user_incar('Relax2D.txt')
        super().__init__(structure, MPScanRelaxSet2D.CONFIG, user_incar_settings=incar, **kwargs)
        self.bandgap = bandgap
        self.kwargs = kwargs

        # self.kwargs.get("user_incar_settings", {
        updates = {}
        # select the KSPACING and smearing parameters based on the bandgap
        if self.bandgap == 0:
            updates["KSPACING"] = 0.22
            updates["SIGMA"] = 0.2
            updates["ISMEAR"] = 2
        else:
            rmin = 25.22 - 1.87 * bandgap  # Eq. 25
            kspacing = 2 * np.pi * 1.0265 / (rmin - 1.0183)  # Eq. 29
            # cap the KSPACING at a max of 0.44, per internal benchmarking
            if kspacing > 0.44:
                kspacing = 0.44
            updates["KSPACING"] = kspacing
            updates["ISMEAR"] = -5
            updates["SIGMA"] = 0.05

        # Don't overwrite things the user has supplied
        if kwargs.get("user_incar_settings", {}).get("KSPACING"):
            del updates["KSPACING"]

        if kwargs.get("user_incar_settings", {}).get("ISMEAR"):
            del updates["ISMEAR"]

        if kwargs.get("user_incar_settings", {}).get("SIGMA"):
            del updates["SIGMA"]
        
        self._config_dict["INCAR"].update(updates)

def wf_bandstructure2D(structure, c=None):

    c = c or {}
    vasp_cmd = c.get("VASP_CMD", VASP_CMD)
    db_file = c.get("DB_FILE", DB_FILE)
    vdw_kernel = c.get("VDW_KERNEL_DIR", VDW_KERNEL_DIR)

    mpr2d = MPScanRelaxSet2D(structure, force_gamma=True, potcar_functional='PBE')
    '''check bandstructure.yaml'''
    wf = get_wf(structure, "bandstructure.yaml", vis=MPScanRelaxSet2D(structure, force_gamma=True, potcar_functional='PBE'), \
                params=[{'vasp_input_set': mpr2d},{},{},{}], common_params={"vasp_cmd": vasp_cmd, "db_file": db_file,}) #"vdw_kernel_dir": vdw_kernel})

    wf = add_common_powerups(wf, c)

    if c.get("SMALLGAP_KPOINT_MULTIPLY", SMALLGAP_KPOINT_MULTIPLY):
        wf = add_small_gap_multiply(wf, 0.5, 5, "static")
        wf = add_small_gap_multiply(wf, 0.5, 5, "nscf")

    if c.get("STABILITY_CHECK", STABILITY_CHECK):
        wf = add_stability_check(wf, fw_name_constraint="structure optimization")

    if c.get("ADD_WF_METADATA", ADD_WF_METADATA):
        wf = add_wf_metadata(wf, structure)

    return wf