#!/bin/bash
#SBATCH -p  par-single
#SBATCH -n 3
#SBATCH -o /home/users/gmpp/logs_concat/Current_%J.out
#SBATCH -e /home/users/gmpp/logs_concat/Current_%J.err
#SBATCH -t 48:00:00
#SBATCH --mem 20000

/home/users/gmpp/miniconda2/envs/phd37/bin/python /home/users/gmpp/phdscripts/xr_tools/concat_simulation_by_year.py $simulation