#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=dependent_async.output

cd $SLURM_SUBMIT_DIR

# FIX: Loading the specific version found: gcc 7.5.0
module load gcc/7.5.0

# The compilation command is correct for your Taskflow path
g++ -std=c++17 -O3 -I/srv/home/eang2/ECE455lab/lab8/taskflow dependent_async.cpp -o dependent_async -pthread

./dependent_async