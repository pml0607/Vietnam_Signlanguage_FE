#!/bin/bash
#SBATCH --job-name=early_no_trans
#SBATCH --partition=dgx-small
#SBATCH --time=72:00:00
#SBATCH --account=ddt_acc23

#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

#SBATCH --output=S3D/logs/%x_%j_%D.out
#SBATCH --error=S3D/logs/%x_%j_%D.err
#SBATCH --export=MASTER_ADDR=localhost

# export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
source /home/21013187/anaconda3/etc/profile.d/conda.sh
squeue --me
cd /work/21013187/linh/Vietnam_Signlanguage_FE/S3D
module load python cuda
conda deactivate
conda deactivate
conda deactivate

conda activate py311
export CUDA_VISIBLE_DEVICES=1,3
python --version

torchrun --nproc_per_node=2\
 --rdzv_id=101 \
 /work/21013187/linh/Vietnam_Signlanguage_FE/S3D/train.py \
 --save_name='early_no_s3d'

