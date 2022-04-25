import subprocess
import os

script_path = '/home/users/gmpp/phdscripts/xr_tools/slurm_submission.sh'
simulations = [
    # 'CMIP6.HighResMIP.EC-Earth-Consortium.EC-Earth3P.highresSST-present.r1i1p1f',
    # 'CMIP6.HighResMIP.EC-Earth-Consortium.EC-Earth3P.highresSST-present.r2i1p1f1',
    # 'CMIP6.HighResMIP.EC-Earth-Consortium.EC-Earth3P.highresSST-present.r3i1p1f1',
    # 'CMIP6.HighResMIP.EC-Earth-Consortium.EC-Earth3P-HR.highresSST-present.r2i1p1f1',
    # 'CMIP6.HighResMIP.EC-Earth-Consortium.EC-Earth3P-HR.highresSST-present.r3i1p1f1',
    # 'CMIP6.HighResMIP.ECMWF.ECMWF-IFS-HR.highresSST-present.r1i1p1f1'
    'CMIP6.HighResMIP.CNRM-CERFACS.CNRM-CM6-1-HR.highresSST-present.r1i1p1f2',
    'CMIP6.HighResMIP.CNRM-CERFACS.CNRM-CM6-1.highresSST-present.r1i1p1f2'
]


for sim in simulations:
    print('Running simulation {}'.format(sim))
    os.environ['simulation'] = str(sim)
    subprocess.call(['sbatch', '--export=ALL', script_path])
