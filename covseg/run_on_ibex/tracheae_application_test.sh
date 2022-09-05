#!/bin/bash

#SBATCH -N 1

#SBATCH -J tracheae_seg_no_rim_enhance

#SBATCH -o log/%x.%3a.%A.out

#SBATCH -e log/%x.%3a.%A.err

#SBATCH --time=6:00:00

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=6

#SBATCH --mem=64G

##SBATCH --mail-user=longxi.zhou@kaust.edu.sa

##SBATCH --mail-type=ALL



# activate your conda env

echo "Loading anaconda..."


module purge

module load gcc

module load cuda/10.0.130

source ~/.bashrc


echo "...Anaconda env loaded"

source activate ml

python prediction_and_visualization.py --data_dir /ibex/scratch/projects/c2052/Lung_CAD_NMI/applications/tracheae_seg/codes "$@"

echo "...training function Done"
