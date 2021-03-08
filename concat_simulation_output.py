import xarray as xr
import glob
import os
import sys
from dask.diagnostics import ProgressBar


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def open_and_write_files(files, nchunks, filename):
    print("Opening files")
    da = xr.open_mfdataset(files, chunks={'time': nchunks})
    da = da.load()
    print(da)
    da = da.to_array().isel(variable=0).drop('variable')
    da.name = 'var'
    try:
        os.makedirs(outpath + sim)
    except FileExistsError:
        pass
    print('writing')
    with ProgressBar():
        da.to_netcdf(outpath + sim + '/' + filename)


if __name__ == '__main__':
    print(sys.argv)
    simpath = sys.argv[1]
    sim = simpath.split('/')[-1]
    outpath = '/gws/nopw/j04/upscale/gmpp/convzones/'
    files = [f for f in glob.glob(simpath + "**/SL_*.nc", recursive=True)]
    for idx, chunk_files in enumerate(chunks(files, 600)):
        open_and_write_files(chunk_files, nchunks=600, filename='partial_{0:03d}.nc'.format(idx))
        print(f'Done {100 * idx*len(files)/600}%')


