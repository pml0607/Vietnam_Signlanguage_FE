#!/bin/bash
#SBATCH --job-name=early_no_trans
#SBATCH --partition=dgx-small
#SBATCH --time=72:00:00
#SBATCH --account=ddt_acc23

#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

#SBATCH --output=/work/21013187/linh/Vietnam_Signlanguage_FE/Transformer/rgb+landmark/logs/%x_%j_%D.out
#SBATCH --error=/work/21013187/linh/Vietnam_Signlanguage_FE/Transformer/rgb+landmark/logs/%x_%j_%D.err
#SBATCH --export=MASTER_ADDR=localhost

# export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
source /home/21013187/anaconda3/etc/profile.d/conda.sh
squeue --me
cd /work/21013187/linh/Vietnam_Signlanguage_FE/Transformer/rgb+landmark
module load python cuda
conda deactivate
conda deactivate
conda deactivate

conda activate py311
export CUDA_VISIBLE_DEVICES=5
python --version

torchrun --nproc_per_node=1 \
 --rdzv_id=100 \
 --rdzv_backend=c10d \
 /work/21013187/linh/Vietnam_Signlanguage_FE/Transformer/rgb+landmark/train.py \
 --save_name='early_no_trans'

