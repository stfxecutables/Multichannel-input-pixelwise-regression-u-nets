import os
from pathlib import Path

# flake8: noqa
# Use environment variables to auto-detect whether we are running an a Compute Canada cluster:
# Thanks to https://github.com/DM-Berger/unet-learn/blob/master/src/train/load.py for this trick.
COMPUTECANADA = False
IN_COMPUTE_CAN_JOB = False

TMP = os.environ.get("SLURM_TMPDIR")
ACT = os.environ.get("SLURM_ACCOUNT")


if ACT:  # If only ACT is True, we are just in a login node
    COMPUTECANADA = True
if TMP:  # If there is a SLURM_TMPDIR we are (probably) running on a non-login node, i.e. in a job
    COMPUTECANADA = True
    IN_COMPUTE_CAN_JOB = True

# fmt: off
if COMPUTECANADA:
    DATA_ROOT = Path(str(TMP)).resolve() / "work"
    DIFFUSION_INPUT = DATA_ROOT / "input"
    DIFFUSION_LABEL = DATA_ROOT / "label"
    ADNI_SC  = DATA_ROOT / "SC_registration"  # noqa: E221
    ADNI_M06 = DATA_ROOT / "M06_registration"
    ADNI_M12 = DATA_ROOT / "M12_registration"
    ADNI_M24 = DATA_ROOT / "M24_registration"
else:
    DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"
    DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "Diffusion" / "dti_preprocessed"
    DIFFUSION_INPUT = DATA_ROOT / "Diffusion" / "input"
    DIFFUSION_LABEL = DATA_ROOT / "Diffusion" / "label"
    ADNI_SC  = DATA_ROOT / "ADNI" / "SC_registration"
    ADNI_M06 = DATA_ROOT / "ADNI" / "M06_registration"
    ADNI_M12 = DATA_ROOT / "ADNI" / "M12_registration"
    ADNI_M24 = DATA_ROOT / "ADNI" / "M24_registration"
# fmt: on

IMAGESIZE = 128
ADNI_LIST = [ADNI_M24, ADNI_M12, ADNI_M06, ADNI_SC]
