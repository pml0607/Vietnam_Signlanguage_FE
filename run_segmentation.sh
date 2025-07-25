#!/bin/bash
#SBATCH --job-name=segment_yolo
#SBATCH --partition=dgx-small
#SBATCH --time=72:00:00
#SBATCH --account=ddt_acc23

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          
#SBATCH --cpus-per-task=1                

#SBATCH --output=/work/21013187/linh/Vietnam_Signlanguage_FE/seg_logs/%x_%j_%D.out
#SBATCH --error=/work/21013187/linh/Vietnam_Signlanguage_FE/seg_logs/%x_%j_%D.err
#SBATCH --export=MASTER_ADDR=localhost

source /home/21013187/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate py311

cd /work/21013187/linh/Vietnam_Signlanguage_FE/Ultralytics

python -c "import torch; print('Visible GPUs:', torch.cuda.device_count())"
export CUDA_VISIBLE_DEVICES=2,5,7
python add_bg.py
