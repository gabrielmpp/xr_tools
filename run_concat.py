import glob
import subprocess

simpath ='/work/scratch-nompiio/gmpp/'

experiments = glob.glob(simpath + 'experiment*')
logs_path = '/home/users/gmpp/logs/'

python = '/home/users/gmpp/miniconda2/envs/phd37/bin/python'
script_path = '/home/users/gmpp/phdscripts/xr_tools/concat_simulation_output.py'
for experiment in experiments:
    subprocess.call(['bsub',
                     '-o', logs_path + '%J.out',
                     '-e', logs_path + '%J.err',
                     '-W', '48:00',
                     '-R', 'rusage[mem=10000]',
                     '-M', '140000',
                     '-n', '10',
                     '-q', 'par-multi',
                     python, script_path, experiment]
                    )

