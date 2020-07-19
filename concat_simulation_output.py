import xarray as xr
import glob
import os
import sys

simpath = sys.argv[1]

sim = simpath.split('/')[-1]
outpath = '/group_workspaces/jasmin4/upscale/gmpp/convzones/'
files = [f for f in glob.glob(simpath + "**/*.nc", recursive=True)]
print(files)
da = xr.open_mfdataset(files).to_array().isel(variable=0).drop('variable')
# da = da.chunk({'time': 100})
try:
    os.makedirs(outpath + sim)
except FileExistsError:
    pass

da = da.to_netcdf(outpath + sim + '/full_array.nc')