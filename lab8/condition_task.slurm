#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=condition_task.output

cd $SLURM_SUBMIT_DIR
module load gcc

g++ -std=c++17 -O3 -I/srv/home/eang2/ECE455lab/lab8/taskflow \condition_task.cpp -o condition_task

./condition_task


#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=condition_task.output

cd $SLURM_SUBMIT_DIR

# IMPORTANT: Use the full version if 'module load gcc' fails
module load gcc

# NOTE: Removed the backslash and ensured a space before the source file
g++ -std=c++17 -O3 -I/srv/home/eang2/ECE455lab/lab8/taskflow condition_task.cpp -o condition_task

./condition_task
