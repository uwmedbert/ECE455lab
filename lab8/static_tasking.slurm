#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=static_tasking.output

cd $SLURM_SUBMIT_DIR

module load gcc/14.3.0

g++ -std=c++17 -O3 -pthread -I taskflow -o dependent_async

./static_tasking