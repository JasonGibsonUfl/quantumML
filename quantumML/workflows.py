from pymatgen.io.vasp.sets import MPRelaxSet
from atomate.vasp.workflows.base.core import get_wf
from atomate.vasp.powerups import add_small_gap_multiply, add_stability_check, add_modify_incar, \
    add_wf_metadata, add_common_powerups

VASP_CMD = ">>vasp_cmd<<"
DB_FILE = ">>db_file<<"
SMALLGAP_KPOINT_MULTIPLY = True
STABILITY_CHECK = False
ADD_WF_METADATA = True

def wf_bandstructure2D(structure, c=None):

    c = c or {}
    vasp_cmd = c.get("VASP_CMD", VASP_CMD)
    db_file = c.get("DB_FILE", DB_FILE)
    '''check bandstructure.yaml'''
    wf = get_wf(structure, "bandstructure.yaml", vis=MPRelaxSet(structure, force_gamma=True),
                common_params={"vasp_cmd": vasp_cmd, "db_file": db_file})

    wf = add_common_powerups(wf, c)

    if c.get("SMALLGAP_KPOINT_MULTIPLY", SMALLGAP_KPOINT_MULTIPLY):
        wf = add_small_gap_multiply(wf, 0.5, 5, "static")
        wf = add_small_gap_multiply(wf, 0.5, 5, "nscf")

    if c.get("STABILITY_CHECK", STABILITY_CHECK):
        wf = add_stability_check(wf, fw_name_constraint="structure optimization")

    if c.get("ADD_WF_METADATA", ADD_WF_METADATA):
        wf = add_wf_metadata(wf, structure)

    return wf