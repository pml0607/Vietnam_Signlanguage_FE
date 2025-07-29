#!/bin/bash
#SBATCH --job-name=watchers
#SBATCH --partition=dgx-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --account=ddt_acc23
#SBATCH --output=watch_logs/%x_%j.out
#SBATCH --error=watch_logs/%x_%j.err

source /home/21013187/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate py311
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/work/21013187/linh/Vietnam_Signlanguage_FE:$PYTHONPATH
# Watcher Landmark → GPU 0
CUDA_VISIBLE_DEVICES=0 WATCHER_TYPE=landmark python Source/run_watcher.py &

# Watcher Inference → GPU 1
CUDA_VISIBLE_DEVICES=1 WATCHER_TYPE=inference python Source/run_watcher.py &

wait
