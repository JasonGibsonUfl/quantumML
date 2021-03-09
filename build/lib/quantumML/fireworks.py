from fireworks import Firework
from atomate.vasp.config import (
    HALF_KPOINTS_FIRST_RELAX,
    RELAX_MAX_FORCE,
    VASP_CMD,
    DB_FILE,
    VDW_KERNEL_DIR,
)
from pymatgen.io.vasp.sets import MPRelaxSet
import warnings
from atomate.vasp.firetasks.write_inputs import WriteVaspFromIOSet, WriteVaspStaticFromPrev, WriteVaspNSCFFromPrev
from atomate.vasp.firetasks.run_calc import RunVaspCustodian
from atomate.common.firetasks.glue_tasks import PassCalcLocs, CopyFiles
from atomate.vasp.firetasks.glue_tasks import CopyVaspOutputs
from atomate.vasp.firetasks.parse_outputs import VaspToDb
from pymatgen.io.vasp.sets import MPStaticSet

class NonSCFFW2D(Firework):
    def __init__(
        self,
        parents=None,
        prev_calc_dir=None,
        structure=None,
        name="nscf",
        mode="uniform",
        vasp_cmd=VASP_CMD,
        copy_vasp_outputs=True,
        db_file=DB_FILE,
        vdw_kernel_dir=VDW_KERNEL_DIR,
        input_set_overrides=None,
        **kwargs
    ):
        """
        Standard NonSCF Calculation Firework supporting uniform and line modes.
        Args:
            structure (Structure): Input structure - used only to set the name
                of the FW.
            name (str): Name for the Firework.
            mode (str): "uniform" or "line" mode.
            vasp_cmd (str): Command to run vasp.
            copy_vasp_outputs (bool): Whether to copy outputs from previous
                run. Defaults to True.
            prev_calc_dir (str): Path to a previous calculation to copy from
            db_file (str): Path to file specifying db credentials.
            parents (Firework): Parents of this particular Firework.
                FW or list of FWS.
            input_set_overrides (dict): Arguments passed to the
                "from_prev_calc" method of the MPNonSCFSet. This parameter
                allows a user to modify the default values of the input set.
                For example, passing the key value pair
                    {'reciprocal_density': 1000}
                will override default k-point meshes for uniform calculations.
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """
        input_set_overrides = input_set_overrides or {}

        fw_name = "{}-{} {}".format(
            structure.composition.reduced_formula if structure else "unknown",
            name,
            mode,
        )
        t = []

        if prev_calc_dir:
            t.append(
                CopyVaspOutputs(calc_dir=prev_calc_dir, additional_files=["CHGCAR"])
            )
        elif parents:
            t.append(CopyVaspOutputs(calc_loc=True, additional_files=["CHGCAR"]))
        else:
            raise ValueError("Must specify previous calculation for NonSCFFW")

        mode = mode.lower()
        if mode == "uniform":
            t.append(
                WriteVaspNSCFFromPrev(
                    prev_calc_dir=".", mode="uniform", **input_set_overrides
                )
            )
        else:
            t.append(
                WriteVaspNSCFFromPrev(
                    prev_calc_dir=".", mode="line", **input_set_overrides
                )
            )
        t.append(CopyFiles(from_dir=vdw_kernel_dir))
        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, auto_npar=">>auto_npar<<"))
        t.append(PassCalcLocs(name=name))
        t.append(
            VaspToDb(
                db_file=db_file,
                additional_fields={"task_label": name + " " + mode},
                parse_dos=(mode == "uniform"),
                bandstructure_mode=mode,
            )
        )

        super(NonSCFFW2D, self).__init__(t, parents=parents, name=fw_name, **kwargs)

class OptimizeFW2D(Firework):
    def __init__(
            self,
            structure,
            name="structure optimization",
            vasp_input_set=None,
            vasp_cmd=VASP_CMD,
            override_default_vasp_params=None,
            ediffg=None,
            db_file=DB_FILE,
            vdw_kernel_dir=VDW_KERNEL_DIR,
            force_gamma=True,
            job_type="double_relaxation_run",
            max_force_threshold=RELAX_MAX_FORCE,
            auto_npar=">>auto_npar<<",
            half_kpts_first_relax=HALF_KPOINTS_FIRST_RELAX,
            parents=None,
            **kwargs
    ):
        """
        Optimize the given structure.
        Args:
            structure (Structure): Input structure.
            name (str): Name for the Firework.
            vasp_input_set (VaspInputSet): input set to use. Defaults to MPRelaxSet() if None.
            override_default_vasp_params (dict): If this is not None, these params are passed to
                the default vasp_input_set, i.e., MPRelaxSet. This allows one to easily override
                some settings, e.g., user_incar_settings, etc.
            vasp_cmd (str): Command to run vasp.
            ediffg (float): Shortcut to set ediffg in certain jobs
            db_file (str): Path to file specifying db credentials to place output parsing.
            force_gamma (bool): Force gamma centered kpoint generation
            job_type (str): custodian job type (default "double_relaxation_run")
            max_force_threshold (float): max force on a site allowed at end; otherwise, reject job
            auto_npar (bool or str): whether to set auto_npar. defaults to env_chk: ">>auto_npar<<"
            half_kpts_first_relax (bool): whether to use half the kpoints for the first relaxation
            parents ([Firework]): Parents of this particular Firework.
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """
        override_default_vasp_params = override_default_vasp_params or {}
        vasp_input_set = vasp_input_set or MPRelaxSet(
            structure, force_gamma=force_gamma, **override_default_vasp_params
        )

        if vasp_input_set.incar["ISIF"] in (0, 1, 2, 7) and job_type == "double_relaxation":
            warnings.warn(
                "A double relaxation run might not be appropriate with ISIF {}".format(
                    vasp_input_set.incar["ISIF"]))

        t = []
        t.append(WriteVaspFromIOSet(structure=structure, vasp_input_set=vasp_input_set))
        t.append(CopyFiles(from_dir=vdw_kernel_dir))
        t.append(
            RunVaspCustodian(
                vasp_cmd=vasp_cmd,
                job_type=job_type,
                max_force_threshold=max_force_threshold,
                ediffg=ediffg,
                auto_npar=auto_npar,
                half_kpts_first_relax=half_kpts_first_relax,
            )
        )
        t.append(PassCalcLocs(name=name))
        t.append(VaspToDb(db_file=db_file, additional_fields={"task_label": name}))
        super(OptimizeFW2D, self).__init__(
            t,
            parents=parents,
            name="{}-{}".format(structure.composition.reduced_formula, name),
            **kwargs
        )

class StaticFW2D(Firework):
    def __init__(
        self,
        structure=None,
        name="static",
        vasp_input_set=None,
        vasp_input_set_params=None,
        vasp_cmd=VASP_CMD,
        prev_calc_loc=True,
        prev_calc_dir=None,
        db_file=DB_FILE,
        vdw_kernel_dir=VDW_KERNEL_DIR,
        vasptodb_kwargs=None,
        parents=None,
        **kwargs
    ):
        """
        Standard static calculation Firework - either from a previous location or from a structure.
        Args:
            structure (Structure): Input structure. Note that for prev_calc_loc jobs, the structure
                is only used to set the name of the FW and any structure with the same composition
                can be used.
            name (str): Name for the Firework.
            vasp_input_set (VaspInputSet): input set to use (for jobs w/no parents)
                Defaults to MPStaticSet() if None.
            vasp_input_set_params (dict): Dict of vasp_input_set kwargs.
            vasp_cmd (str): Command to run vasp.
            prev_calc_loc (bool or str): If true (default), copies outputs from previous calc. If
                a str value, retrieves a previous calculation output by name. If False/None, will create
                new static calculation using the provided structure.
            prev_calc_dir (str): Path to a previous calculation to copy from
            db_file (str): Path to file specifying db credentials.
            parents (Firework): Parents of this particular Firework. FW or list of FWS.
            vasptodb_kwargs (dict): kwargs to pass to VaspToDb
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """
        t = []

        vasp_input_set_params = vasp_input_set_params or {}
        vasptodb_kwargs = vasptodb_kwargs or {}
        if "additional_fields" not in vasptodb_kwargs:
            vasptodb_kwargs["additional_fields"] = {}
        vasptodb_kwargs["additional_fields"]["task_label"] = name

        fw_name = "{}-{}".format(
            structure.composition.reduced_formula if structure else "unknown", name
        )

        if prev_calc_dir:
            t.append(CopyVaspOutputs(calc_dir=prev_calc_dir, contcar_to_poscar=True))
            t.append(WriteVaspStaticFromPrev(other_params=vasp_input_set_params))
        elif parents:
            if prev_calc_loc:
                t.append(
                    CopyVaspOutputs(calc_loc=prev_calc_loc, contcar_to_poscar=True)
                )
            t.append(WriteVaspStaticFromPrev(other_params=vasp_input_set_params))
        elif structure:
            vasp_input_set = vasp_input_set or MPStaticSet(
                structure, **vasp_input_set_params
            )
            t.append(
                WriteVaspFromIOSet(structure=structure, vasp_input_set=vasp_input_set)
            )
        else:
            raise ValueError("Must specify structure or previous calculation")
        t.append(CopyFiles(from_dir=vdw_kernel_dir))
        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, auto_npar=">>auto_npar<<"))
        t.append(PassCalcLocs(name=name))
        t.append(VaspToDb(db_file=db_file, **vasptodb_kwargs))
        super(StaticFW2D, self).__init__(t, parents=parents, name=fw_name, **kwargs)
