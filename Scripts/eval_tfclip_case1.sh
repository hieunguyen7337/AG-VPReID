#!/bin/bash -l
#PBS -N tfclip_eval_case1
#PBS -l walltime=10:00:00
#PBS -l mem=32gb
#PBS -l ncpus=4
#PBS -l ngpus=1
#PBS -j eo
#PBS -m abe

echo '================================================'
echo "CWD = ${PBS_O_WORKDIR}"
echo '================================================'
cd "$PBS_O_WORKDIR"

echo '=========='
echo 'Load CUDA & cuDNN modules'
echo '=========='
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

echo '=========='
echo 'Activate conda env'
echo '=========='
# Robust conda activation across PBS environments
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfclip

# Helpful runtime info (GPU status and library versions)
nvidia-smi || true
which python
python -c "import torch, torchvision; print('PyTorch', torch.__version__, 'Torchvision', torchvision.__version__)"

# Respect a GPU index passed as arg (defaults to GPU 0)
GPU_INDEX="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"

echo '========================='
echo 'Navigate to baseline directory'
echo '========================='
cd baselines/tfclip || { echo "Failed to find baselines/tfclip directory" ; exit 1; }

echo '========================='
echo 'Run Evaluation (Case 1: Aerial to Ground)'
echo '========================='
python eval_main.py --custom_output_dir "results/case1_aerial_to_ground" --output_dir "logs/all"

echo 'Done.'
