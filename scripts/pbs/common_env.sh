# common_env.sh -- sourced by every PBS job. Edit the paths once, here.
module purge
module load python/3.9.13-gcc-jhn          # the venv's python symlink points at this module: it MUST be loaded
module load cuda/12.4.0-gcc-ow5
export PYTHONNOUSERSITE=1
source $HOME/pss_env/bin/activate

export ROOT=$HOME/pss/pss_revision_code_results_20261019
export MODELS=$ROOT/Models
export SPLITS=$ROOT/Config/data_splits_heldout.json
export RUNS=$ROOT/runs_multiseed
export UQ=$ROOT/uq
export UQS=$ROOT/uq_single

# full standardised BraTS 2021 cohort (train+val+test); also used for the 251-case test evaluation
export FULL_DIR=$HOME/data/BRATS2021_standardized
export TEST_DIR=$FULL_DIR
# BraTS-Africa: RAW root (input to preprocessing) and the STANDARDISED output (flat <pid>/<pid>_<mod>.nii.gz)
export AFRICA_RAW=$HOME/data/BRATS_Africa
export AFRICA_DIR=$HOME/data/BRATS_Africa_standardized
# the 145 cases analysed in the paper; evaluate Africa with --split test against this file (NOT --africa)
export AFRICA_SPLITS=$ROOT/Config/africa_145_splits.json

mkdir -p $ROOT/logs $UQ $UQS $RUNS
cd $MODELS
