#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=condition_task.output

cd $SLURM_SUBMIT_DIR

# Use the newest compatible compiler for Euler Generation 8
module purge
module load gcc/14.3.0

# Compile the Taskflow program
g++ condition_task.cpp \
    -std=c++17 -O2 -pthread \
    -I/srv/home/eang2/ECE455lab/lab8/taskflow/taskflow \
    -o condition_task

# Run the executable
./condition_task
