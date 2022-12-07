#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=05-00:00
#SBATCH --mem-per-cpu=6G
#SBATCH --array=1-28

module purge
module load python/3.7 gcc/11.3.0 r/4.2.1
source /home/jswyou/projects/def-quiltyjo/jswyou/COMBO_ENV/bin/activate

echo "Starting run at: `date`"

echo "Starting task $SLURM_ARRAY_TASK_ID"

DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" dir_names.txt)
mkdir $DIR && cd $DIR
python ../main.py

echo "Program finished with exit code $? at: `date`"
