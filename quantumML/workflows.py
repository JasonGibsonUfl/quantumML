from pymatgen.io.vasp.sets import DictSet, _load_yaml_config
from pymatgen.core.structure import Structure
from atomate.vasp.workflows.base.core import get_wf
from atomate.vasp.powerups import add_small_gap_multiply, add_stability_check, add_modify_incar, \
    add_wf_metadata, add_common_powerups


from atomate.vasp.config import SMALLGAP_KPOINT_MULTIPLY, STABILITY_CHECK, VASP_CMD, DB_FILE, \
    ADD_WF_METADATA

from pymatgen.io.vasp.inputs import Incar, Poscar, Potcar, Kpoints, VaspInput
from pymatgen.io.vasp.outputs import Vasprun, Outcar

from typing import Optional
import warnings
import glob
import os
from monty.serialization import loadfn
from pathlib import Path

module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

def _read_user_incar(fname):
    fpath = os.path.join(module_dir,'library',fname)
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
        incar = _read_user_incar('Relax2D')
        super().__init__(structure, MPRelaxSet2D.CONFIG, user_incar_settings={incar}, **kwargs)
        self.kwargs = kwargs


class MPStaticSet2D(MPRelaxSet2D):
    """
    Creates input files for a static calculation.
    """

    def __init__(
            self,
            structure,
            prev_incar=None,
            prev_kpoints=None,
            lepsilon=False,
            lcalcpol=False,
            reciprocal_density=100,
            small_gap_multiply=None,
            **kwargs
    ):
        """
        Args:
            structure (Structure): Structure from previous run.
            prev_incar (Incar): Incar file from previous run.
            prev_kpoints (Kpoints): Kpoints from previous run.
            lepsilon (bool): Whether to add static dielectric calculation
            reciprocal_density (int): For static calculations, we usually set the
                reciprocal density by volume. This is a convenience arg to change
                that, rather than using user_kpoints_settings. Defaults to 100,
                which is ~50% more than that of standard relaxation calculations.
            small_gap_multiply ([float, float]): If the gap is less than
                1st index, multiply the default reciprocal_density by the 2nd
                index.
            **kwargs: kwargs supported by MPRelaxSet.
        """
        super().__init__(structure, **kwargs)
        if isinstance(prev_incar, str):
            prev_incar = Incar.from_file(prev_incar)
        if isinstance(prev_kpoints, str):
            prev_kpoints = Kpoints.from_file(prev_kpoints)

        self.prev_incar = prev_incar
        self.prev_kpoints = prev_kpoints
        self.reciprocal_density = reciprocal_density
        self.kwargs = kwargs
        self.lepsilon = lepsilon
        self.lcalcpol = lcalcpol
        self.small_gap_multiply = small_gap_multiply

    @property
    def incar(self):
        """
        :return: Incar
        """
        parent_incar = super().incar
        incar = (
            Incar(self.prev_incar)
            if self.prev_incar is not None
            else Incar(parent_incar)
        )

        incar.update(
            {
                "IBRION": -1,
                "ISMEAR": -5,
                "LAECHG": True,
                "LCHARG": True,
                "LORBIT": 11,
                "LVHAR": True,
                "LWAVE": False,
                "NSW": 0,
                "ICHARG": 0,
                "ALGO": "Normal",
            }
        )

        if self.lepsilon:
            incar["IBRION"] = 8
            incar["LEPSILON"] = True

            # LPEAD=T: numerical evaluation of overlap integral prevents
            # LRF_COMMUTATOR errors and can lead to better expt. agreement
            # but produces slightly different results
            incar["LPEAD"] = True

            # Note that DFPT calculations MUST unset NSW. NSW = 0 will fail
            # to output ionic.
            incar.pop("NSW", None)
            incar.pop("NPAR", None)

        if self.lcalcpol:
            incar["LCALCPOL"] = True

        for k in ["MAGMOM", "NUPDOWN"] + list(
                self.kwargs.get("user_incar_settings", {}).keys()
        ):
            # For these parameters as well as user specified settings, override
            # the incar settings.
            if parent_incar.get(k, None) is not None:
                incar[k] = parent_incar[k]
            else:
                incar.pop(k, None)

        # use new LDAUU when possible b/c the Poscar might have changed
        # representation
        if incar.get("LDAU"):
            u = incar.get("LDAUU", [])
            j = incar.get("LDAUJ", [])
            if sum([u[x] - j[x] for x, y in enumerate(u)]) > 0:
                for tag in ("LDAUU", "LDAUL", "LDAUJ"):
                    incar.update({tag: parent_incar[tag]})
            # ensure to have LMAXMIX for GGA+U static run
            if "LMAXMIX" not in incar:
                incar.update({"LMAXMIX": parent_incar["LMAXMIX"]})

        # Compare ediff between previous and staticinputset values,
        # choose the tighter ediff
        incar["EDIFF"] = min(incar.get("EDIFF", 1), parent_incar["EDIFF"])
        return incar

    @property
    def kpoints(self) -> Optional[Kpoints]:
        """
        :return: Kpoints
        """
        self._config_dict["KPOINTS"]["reciprocal_density"] = self.reciprocal_density
        kpoints = super().kpoints

        # Prefer to use k-point scheme from previous run
        # except for when lepsilon = True is specified
        if kpoints is not None:
            if self.prev_kpoints and self.prev_kpoints.style != kpoints.style:
                if (self.prev_kpoints.style == Kpoints.supported_modes.Monkhorst) and (
                        not self.lepsilon
                ):
                    k_div = [kp + 1 if kp % 2 == 1 else kp for kp in kpoints.kpts[0]]
                    kpoints = Kpoints.monkhorst_automatic(k_div)
                else:
                    kpoints = Kpoints.gamma_automatic(kpoints.kpts[0])
        return kpoints

    def override_from_prev_calc(self, prev_calc_dir="."):
        """
        Update the input set to include settings from a previous calculation.
        Args:
            prev_calc_dir (str): The path to the previous calculation directory.
        Returns:
            The input set with the settings (structure, k-points, incar, etc)
            updated using the previous VASP run.
        """
        vasprun, outcar = get_vasprun_outcar(prev_calc_dir)

        self.prev_incar = vasprun.incar
        self.prev_kpoints = vasprun.kpoints

        if self.standardize:
            warnings.warn(
                "Use of standardize=True with from_prev_run is not "
                "recommended as there is no guarantee the copied "
                "files will be appropriate for the standardized "
                "structure."
            )

        self._structure = get_structure_from_prev_run(vasprun, outcar)

        # multiply the reciprocal density if needed
        if self.small_gap_multiply:
            gap = vasprun.eigenvalue_band_properties[0]
            if gap <= self.small_gap_multiply[0]:
                self.reciprocal_density = (
                        self.reciprocal_density * self.small_gap_multiply[1]
                )

        return self

    @classmethod
    def from_prev_calc(cls, prev_calc_dir, **kwargs):
        """
        Generate a set of Vasp input files for static calculations from a
        directory of previous Vasp run.
        Args:
            prev_calc_dir (str): Directory containing the outputs(
                vasprun.xml and OUTCAR) of previous vasp run.
            **kwargs: All kwargs supported by MPStaticSet, other than prev_incar
                and prev_structure and prev_kpoints which are determined from
                the prev_calc_dir.
        """
        input_set = cls(_dummy_structure, **kwargs)
        return input_set.override_from_prev_calc(prev_calc_dir=prev_calc_dir)
    
    
def get_vasprun_outcar(path, parse_dos=True, parse_eigen=True):
    """
    :param path: Path to get the vasprun.xml and OUTCAR.
    :param parse_dos: Whether to parse dos. Defaults to True.
    :param parse_eigen: Whether to parse eigenvalue. Defaults to True.
    :return:
    """
    path = Path(path)
    vruns = list(glob.glob(str(path / "vasprun.xml*")))
    outcars = list(glob.glob(str(path / "OUTCAR*")))

    if len(vruns) == 0 or len(outcars) == 0:
        raise ValueError(
            "Unable to get vasprun.xml/OUTCAR from prev calculation in %s" % path
        )
    vsfile_fullpath = str(path / "vasprun.xml")
    outcarfile_fullpath = str(path / "OUTCAR")
    vsfile = vsfile_fullpath if vsfile_fullpath in vruns else sorted(vruns)[-1]
    outcarfile = (
        outcarfile_fullpath if outcarfile_fullpath in outcars else sorted(outcars)[-1]
    )
    return (
        Vasprun(vsfile, parse_dos=parse_dos, parse_eigen=parse_eigen),
        Outcar(outcarfile),
    )

def get_structure_from_prev_run(vasprun, outcar=None):
    """
    Process structure from previous run.
    Args:
        vasprun (Vasprun): Vasprun that contains the final structure
            from previous run.
        outcar (Outcar): Outcar that contains the magnetization info from
            previous run.
    Returns:
        Returns the magmom-decorated structure that can be passed to get
        Vasp input files, e.g. get_kpoints.
    """
    structure = vasprun.final_structure

    site_properties = {}
    # magmom
    if vasprun.is_spin:
        if outcar and outcar.magnetization:
            site_properties.update({"magmom": [i["tot"] for i in outcar.magnetization]})
        else:
            site_properties.update({"magmom": vasprun.parameters["MAGMOM"]})
    # ldau
    if vasprun.parameters.get("LDAU", False):
        for k in ("LDAUU", "LDAUJ", "LDAUL"):
            vals = vasprun.incar[k]
            m = {}
            l_val = []
            s = 0
            for site in structure:
                if site.specie.symbol not in m:
                    m[site.specie.symbol] = vals[s]
                    s += 1
                l_val.append(m[site.specie.symbol])
            if len(l_val) == len(structure):
                site_properties.update({k.lower(): l_val})
            else:
                raise ValueError(
                    "length of list {} not the same as" "structure".format(l_val)
                )

    return structure.copy(site_properties=site_properties)

_dummy_structure = Structure(
    [1, 0, 0, 0, 1, 0, 0, 0, 1],
    ["I"],
    [[0, 0, 0]],
    site_properties={"magmom": [[0, 0, 1]]},
)
def wf_bandstructure2D(structure, c=None):

    c = c or {}
    vasp_cmd = c.get("VASP_CMD", VASP_CMD)
    db_file = c.get("DB_FILE", DB_FILE)
    mpr2d = MPRelaxSet2D(structure, force_gamma=True)
    mps2d = MPStaticSet2D(structure)
    '''check bandstructure.yaml'''
    wf = get_wf(structure, "bandstructure.yaml", vis=MPRelaxSet2D(structure, force_gamma=True), \
                params=[{'vasp_input_set': mpr2d}], common_params={"vasp_cmd": vasp_cmd, "db_file": db_file})

    wf = add_common_powerups(wf, c)

    if c.get("SMALLGAP_KPOINT_MULTIPLY", SMALLGAP_KPOINT_MULTIPLY):
        wf = add_small_gap_multiply(wf, 0.5, 5, "static")
        wf = add_small_gap_multiply(wf, 0.5, 5, "nscf")

    if c.get("STABILITY_CHECK", STABILITY_CHECK):
        wf = add_stability_check(wf, fw_name_constraint="structure optimization")

    if c.get("ADD_WF_METADATA", ADD_WF_METADATA):
        wf = add_wf_metadata(wf, structure)

    return wf