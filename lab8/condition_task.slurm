#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=condition_task.output

cd $SLURM_SUBMIT_DIR
module load gcc/11.3.0

g++ condition_task.cpp -I/srv/home/eang2/ECE455lab/lab8/taskflow -std=c++17 -O2 -o condition_task

./condition_task
