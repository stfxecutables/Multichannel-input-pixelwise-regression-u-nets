import os
from pathlib import Path

# Some code is adapted from: https://github.com/DM-Berger/unet-learn/blob/master/src/train/load.py
COMPUTECANADA = False  # are we running on Compute Canada
IN_COMPUTE_CAN_JOB = False  # are we running inside a Compute Canada Job

TMP = os.environ.get("SLURM_TMPDIR")
ACT = os.environ.get("SLURM_ACCOUNT")


if ACT:  # we are on Compute Canada, but not in a job script, so we don't want to run too much
    COMPUTECANADA = True
if TMP:  # running inside Compute Canada
    COMPUTECANADA = True
    IN_COMPUTE_CAN_JOB = True

if COMPUTECANADA:
    # fmt: off
    DATA_ROOT = Path(str(TMP)).resolve() / "work"
    DIFFUSION_INPUT = DATA_ROOT / "input"
    DIFFUSION_LABEL = DATA_ROOT / "label"
    ADNI_SC  = DATA_ROOT / "SC_registration"
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
